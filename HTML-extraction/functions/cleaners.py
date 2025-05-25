import re
import os
import json
import yaml
import random
from tqdm import tqdm
from bs4 import BeautifulSoup as BS
import trafilatura
from . import meta_retriever
from . import _checker_functions
from . import utils


class Cleaner():
    """Generic class for cleaning html data."""

    # ++++++GETTERS AND SETTERS++++++
    @property
    def src_dir(self):
        return self._src_dir

    @src_dir.setter
    def src_dir(self, value):
        """Setter for the source directory with checks."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError("Source directory must be a string.")
            if not os.path.isdir(value):
                raise ValueError("Source directory does not exist.")
        self._src_dir = value

    @property
    def goal_dir(self):
        return self._goal_dir

    @goal_dir.setter
    def goal_dir(self, value):
        """Setter for the goal directory with checks and option to create
        a new directory."""
        if not isinstance(value, str) and value is not None:
            raise ValueError("Goal directory must be a string.")
        if value is not None and not os.path.isdir(value):
            print("Goal directory does not exist or is not provided.")
            create = input("Do you want to create it? (y/n) ") == "y"
            if create:
                os.makedirs(value)
                print(f"Directory {value} created.")
            else:
                print("In this case, please provide an existing directory.")
        elif value is None:
            pass
        self._goal_dir = value

    @property
    def config_section(self):
        return self._config_section

    @config_section.setter
    def config_section(self, value):
        self._config_section = value

    @property
    def parallelize(self):
        return self._parallelize

    @parallelize.setter
    def parallelize(self, value):
        if not isinstance(value, bool):
            raise ValueError("Parallelize must be a boolean.")
        self._parallelize = value

    @property
    def regen_existing(self):
        return self._regen_existing

    @regen_existing.setter
    def regen_existing(self, value):
        if not isinstance(value, bool):
            raise ValueError("Regen_existing must be a boolean.")
        self._regen_existing = value

    # ++++++METHODS++++++
    def __init__(self, src_dir=None, goal_dir=None, config_section=None):
        """Constructor for the Cleaner class. Source directory, goal directory,
        and config section can be provided as arguments, but can also be set
        later."""
        self.src_dir = src_dir
        self.goal_dir = goal_dir
        self.config_section = config_section
        self._parallelize = False
        self._regen_existing = False

    def gate_tests(self):
        """Checks if necessary parameters are provided to start cleaning."""
        if not self.src_dir:
            raise ValueError("Please provide a source directory.")
        if not self.goal_dir:
            raise ValueError("Please provide a goal directory.")
        if not self.config_section:
            raise ValueError("Please provide a config section.")
        if self.src_dir == self.goal_dir:
            raise ValueError("Source and goal directories must be different.")
        return True

    def read_config_section(
        self, config_file='config.yaml',
        subsections=(
            'irrelevant_subdomains', 'metadata_keep', 'checker_functions'
        )
    ):
        """Reads the config file and extracts the relevant section."""
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        config_params = config[self.config_section]
        if all((key in config_params for key in subsections)):
            for key, value in config_params.items():
                if value is None:
                    config_params[key] = []
            try:
                for func in config_params['checker_functions']:
                    func = getattr(_checker_functions, func)
            except AttributeError as excep:
                raise AttributeError(
                    "One or more functions in the 'checker_functions' list "
                    "do not exist in the 'checker_functions' module."
                ) from excep
            return config_params

        raise KeyError(
            "Config section is not properly formatted. Make sure it contains "
            "the keys 'irrelevant_subdomains', 'metadata_keep', "
            "'checker_functions', and 'other_info_funcs."
        )


class ArticleCleaner(Cleaner):
    """Class for cleaning html data from article formats."""

    # ++++++GETTERS AND SETTERS++++++
    @property
    def meta_retriever(self):
        return self._meta_retriever

    @meta_retriever.setter
    def meta_retriever(self, value):
        if (
            not isinstance(value, meta_retriever.MetaRetriever)
            and value is not None
        ):
            raise ValueError("MetaRetriever must be an instance of the "
                             "MetaRetriever class.")
        self._meta_retriever = value

    # ++++++METHODS++++++
    def __init__(
        self, src_dir=None, goal_dir=None, config_section=None,
        meta_retriever=None
    ):
        super().__init__(src_dir, goal_dir, config_section)
        self._meta_retriever = meta_retriever

    def configure_meta_retriever(self, config):
        if not self.meta_retriever.labels:
            self.meta_retriever.labels = config['metadata_keep']
        if len(config['other_info_funcs']) > 0:
            self.meta_retriever.other_info_funcs = config['other_info_funcs']

    def clean_articles(self, skip_existing=False):
        config_subsections = (
            'irrelevant_subdomains', 'metadata_keep', 'checker_functions',
            'other_info_funcs'
        )
        self.gate_tests()
        assert self.meta_retriever

        config = self.read_config_section(subsections=config_subsections)
        self.configure_meta_retriever(config)

        files = utils.listdir_filetype(
            self.src_dir, 'json', absolute=True
        )
        if skip_existing:
            goal_files = set(utils.listdir_filetype(
                self.goal_dir, 'json', absolute=True
            ))
            goal_files = {os.path.basename(file) for file in goal_files}
            files = [
                file for file in files
                if os.path.basename(file) not in goal_files
            ]

        for file in tqdm(
            files, desc="Cleaning articles", unit="files",
            total=len(files)
        ):
            self.full_cleaning(file, config)

    def full_cleaning(self, file, config):
        goal_file = os.path.join(
            self.goal_dir, os.path.basename(file)
        )

        if not self.regen_existing and os.path.isfile(goal_file):
            return None

        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            return None

        for subsection in config['irrelevant_subdomains']:
            if re.search(subsection, data['url']):
                return None

        for func in config['checker_functions']:
            func = getattr(_checker_functions, func)
            if not func(data):
                return None

        meta_dict = self.meta_retriever.full_pipeline(
            BS(data['html_content'], 'lxml')
        )
        meta_dict['url'] = data['url']
        meta_dict['time_crawled'] = data['time_crawled']

        text_html = self.trafilatura_article_getter_html(
            data['html_content'], meta_dict=meta_dict
        )
        if text_html is None:
            return None

        text = utils.html_to_text(text_html)

        meta_dict.update({'text_html': text_html, 'text': text})

        with open(goal_file, 'w', encoding='utf-8') as f:
            json.dump(meta_dict, f, ensure_ascii=False, indent=4)

        return None

    def trafilatura_article_getter_md(
        self, html, meta_dict,
        heading_key=None
    ):
        """Uses trafilatura to extract the main text from an article,
        formatted in markdown.
        """

        text = trafilatura.extract(
            html,
            output_format="markdown",
            with_metadata=False,
            include_comments=False,
            include_images=False,
            include_formatting=False,
        )

        if text is None:
            return None

        # Format lists to be correctly recognized by markdown
        text = re.sub(r'(?m)\n(.*?)\n-$', r'\n- \1', text)

        # If there is no markdown heading, add one from the meta_dict
        if text[0] != '#':
            heading = (
                meta_dict.get(heading_key)
                or meta_dict.get('h1')
                or meta_dict.get('title')
            )
            if heading:
                text = f"# {heading}\n\n{text}"

        return text

    def trafilatura_article_getter_html(
        self, html, meta_dict,
        heading_key=None,
        target_language='de'
    ):
        """Uses trafilatura to extract the main text from an article,
        formatted in simple html.
        """

        try:
            html = trafilatura.extract(
                html,
                output_format="html",
                with_metadata=False,
                include_comments=False,
                include_images=False,
                include_tables=False,
                target_language=target_language
            )
        except Exception as excep:
            print(f"Error in trafilatura: {excep}")
            return None

        if html is None:
            return None

        soup = BS(html, 'html.parser')

        if soup is None:
            return None

        # Check if the first element with text! under body is an h element
        text_elements = [
            elem for elem in soup.body.find_all() if elem.get_text().strip()
        ]

        # Get the first element with text
        first_element = text_elements[0]

        if not (first_element.name and first_element.name.startswith('h')):
            heading = (
                meta_dict.get(heading_key)
                or meta_dict.get('h1')
                or meta_dict.get('title')
            )
            if heading:
                new_header = soup.new_tag('h1')
                new_header.string = heading
                soup.body.insert(0, new_header)

        return soup.prettify()

    def print_example_text(self):

        files = utils.listdir_filetype(
            self.src_dir, 'json', absolute=True
        )

        random_file = random.choice(files)

        with open(random_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        html = data['html_content']
        simple_html = self.trafilatura_article_getter_html(html, {})
        text = utils.html_to_text(simple_html)

        print(text)

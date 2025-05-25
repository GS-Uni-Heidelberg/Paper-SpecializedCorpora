import os
import json
from tqdm import tqdm
from bs4 import BeautifulSoup as BS
from . import utils
from dataclasses import dataclass


@dataclass(frozen=True)
class element_info:
    name: str
    text: str

    def __init__(self, element):
        # Use object.__setattr__ to bypass immutability in a frozen dataclass
        object.__setattr__(
            self, 'name', element.name
        )
        object.__setattr__(
            self, 'text', element.get_text(separator=' ', strip=True)
        )

    def __hash__(self):
        return hash((self.name, self.text))

    def __str__(self):
        return f'{self.name}: {self.text}'


def html_dupl_elements(
    datapath, html_retriever=lambda x: x['text_html']
):
    seen_elements = set()
    duplicate_elements = {}

    corpus = []
    for file in tqdm(
        os.listdir(datapath),
        desc='Retrieving text from files'
    ):
        try:
            with open(
                os.path.join(datapath, file), 'r', encoding='utf-8'
            ) as f:
                data = json.load(f)
            corpus.append(html_retriever(data))
        except Exception as e:
            print(f'Error in file {file}: {e}')

    for element in tqdm(
        corpus,
        desc='Checking for duplicate elements'
    ):
        soup = BS(element, 'html.parser')
        for element in soup.find_all():
            elem = element_info(element)
            if elem in seen_elements:
                duplicate_elements[elem] = duplicate_elements.get(
                    elem, 1
                ) + 1
            else:
                seen_elements.add(elem)

    print(f'Inspected elements: {len(seen_elements)}')
    print(f'...out of which {len(duplicate_elements)} are duplicates.')

    return duplicate_elements


def html_remove_dupl(
    datapath,
    duplicate_elements,
    criteria_checker,
    html_retriever=lambda x: x['text_html'],
):

    corpusfiles = os.listdir(datapath)

    for file in tqdm(corpusfiles):
        try:
            filepath = os.path.join(datapath, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            html = html_retriever(data)
            soup = BS(html, 'html.parser')
            for element in soup.find_all():
                elem = element_info(element)
                if criteria_checker(
                    elem.name, elem.text, duplicate_elements.get(elem, 0)
                ):
                    element.decompose()

            data['text_deduped'] = utils.html_to_text(soup.prettify())

            utils.write_json(data, filepath, overwrite=True)
        except (KeyError, json.JSONDecodeError) as e:
            print(f'Error in file {file}: {e}')


def list_removed_elements(
    duplicate_elements,
    criteria_checker
):
    for element, count in duplicate_elements.items():
        if criteria_checker(
            element.name, element.text, count
        ):
            print(element)


def list_kept_elements(
    duplicate_elements,
    criteria_checker
):
    for element, count in duplicate_elements.items():
        if not criteria_checker(
            element.name, element.text, count
        ):
            print(element, '--->', count, sep=' ')



def save_deduped_elements(
    datapath,
    duplicate_elements,
    criteria_checker,
    overwrite_existing=False
):
    _, first_file_data = utils.jsondir_filepath_data_gen(
        datapath, absolute=True
    ).__next__()
    keys = first_file_data.keys()

    deduped_elements_text = [
        element.text for element, count in duplicate_elements.items()
        if criteria_checker(element.name, element.text, count)
    ]
    deduped_elements_text = '\n\n'.join(deduped_elements_text)

    empty_dict = {k: '' for k in keys}
    empty_dict['text_deduped'] = deduped_elements_text

    utils.write_json(
        empty_dict,
        os.path.join(datapath, 'deduped_elements.json'),
        overwrite=overwrite_existing
    )

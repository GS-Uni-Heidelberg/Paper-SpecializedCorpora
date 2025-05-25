import json
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from langdetect import detect, LangDetectException
from langdetect import DetectorFactory  # Ensure deterministic results
import spacy
from . import utils


DetectorFactory.seed = 0


def delete_with_confirmation(
    corpus,
    filename,
    force,
    title_key=None,
):

    if 'deduped' in filename:
        print('Not deleting deduped_elements file!')
        return None

    if force:
        delete = True

    else:
        data = None
        if title_key is not None:
            with open(os.path.join(corpus, filename), 'r') as f:
                data = json.load(f)
        delete = get_deletion_confirmation(
            filename,
            title_key,
            data,
        )

    if delete:
        os.remove(os.path.join(corpus, filename))

    return None


def get_deletion_confirmation(
    file,
    title_key=None,
    data=None,
):
    if title_key is not None and isinstance(data, dict):
        identifier = data[title_key]
    else:
        identifier = file

    choice = input(f'Remove {identifier}? (Y/n): ')

    if choice == 'Y':
        return True
    if choice == 'n':
        return False
    print('Invalid choice. Skipping...')
    return False


def threshold_checker_factory(threshold, below_threshold=True):
    """Creates a function that checks whether a value is above or below a
    threshold.

    Args:
        threshold (int): The threshold value.
        below_threshold (bool): If True, the function will check if the value
            is below the threshold, else it will check if the value is above
            the threshold.
    """

    def below(x):
        return x < threshold

    def above(x):
        return x > threshold

    if below_threshold:
        return below
    else:
        return above


# CHECK FOR SUSPICIOUS TEXT LENGTHS

def get_textlens(
    corpus,
    text_key='text',
):
    """Get the lengths of the text in each file in the corpus.

    Args:
        corpus (str): Path to the corpus directory.
        text_key (str): Key of the text in the JSON file. Default is 'text'.
        You might want to change to 'text_deduped'.
    """

    textlens = []

    for file in os.listdir(corpus):
        with open(os.path.join(corpus, file), 'r') as f:
            data = json.load(f)
        text = data[text_key]
        text_len = len(text)
        textlens.append(text_len)

    return textlens


def plot_textlens(
    corpus,
    text_key='text',
    bins=100,

):
    """Plot the distribution of text lengths in the corpus.

    Args:
        corpus (str): Path to the corpus directory.
        text_key (str): Key of the text in the JSON file. Default is 'text'.
        You might want to change to 'text_deduped'.
        bins (int): Number of bins for the histogram.

    Returns:
        list: A list of the text lengths.
    """

    if not isinstance(corpus, list):
        textlens = get_textlens(corpus, text_key)
    else:
        textlens = corpus

    title = 'Text Lengths',
    xlabel = 'Text Length',
    ylabel = 'Frequency',

    plt.hist(textlens, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()

    return


def print_textlen_stats(
    corpus,
    text_key='text',
):
    """Print statistics about the text lengths in the corpus.

    Args:
        corpus (str): Path to the corpus directory.
        text_key (str): Key of the text in the JSON file. Default is 'text'.
        You might want to change to 'text_deduped'.
    """

    if not isinstance(corpus, list):
        textlens = get_textlens(corpus, text_key)
    else:
        textlens = corpus

    print('Mean:', np.mean(textlens))
    print('Median:', np.median(textlens))
    print('Max:', np.max(textlens))
    print('Min:', np.min(textlens))

    return


def print_textlen_threshold(
    corpus_path,
    threshold,
    below_threshold=True,
    text_key='text',
):
    """Print texts that are above or below a certain threshold.

    Args:
        corpus_path (str): Path to the corpus directory.
        threshold (int): The threshold value. (Texts with exactly this length
            will not be printed.)
        below_threshold (bool): If True, texts below the threshold will be
            printed, else texts above the threshold will be printed.
        text_key (str): Key of the text in the JSON file. Default is 'text'.
            You might want to change to 'text_deduped'.
    """

    textlens = get_textlens(corpus_path, text_key)
    filelist = os.listdir(corpus_path)

    criterion = threshold_checker_factory(
        threshold, below_threshold
    )

    for i, textlen in enumerate(textlens):
        if criterion(textlen):
            with open(
                os.path.join(corpus_path, filelist[i]),
                'r',
            ) as f:
                data = json.load(f)
            print(filelist[i].upper())
            print()
            print(data[text_key])
            print()
            print(f'Length: {textlen}')
            print()
            print('+'*30)
            print()

    return


def remove_textlen_threshold(
    corpus_path,
    threshold,
    below_threshold=True,
    text_key='text',
    force=False,
    title_key=None,
):
    """Remove texts that are above or below a certain threshold.

    Args:
        corpus_path (str): Path to the corpus directory.
        threshold (int): The threshold value. (Texts with exactly this length
            will not be removed.)
        below_threshold (bool): If True, texts below the threshold will be
            removed, else texts above the threshold will be removed.
        text_key (str): Key of the text in the JSON file. Default is 'text'.
            You might want to change to 'text_deduped'.
        force (bool): If True, the function will remove the files without
            asking for confirmation. Default is False.
        title_key (str): If not None, the function will print the value of this
            key in the data when asking for confirmation. Default is None.
    """

    textlens = get_textlens(corpus_path, text_key)
    filelist = os.listdir(corpus_path)

    criterion = threshold_checker_factory(
        threshold, below_threshold
    )

    for i, textlen in enumerate(textlens):
        if criterion(textlen):
            filename = filelist[i]
            delete_with_confirmation(
                corpus_path,
                filename,
                force,
                title_key,
            )

    return


# CHECK FOR TEXTS WITH MANY SHORT PARAGRAPHS (PROBABLY NOT TEXT)


def get_parlens(
    corpus,
    text_key='text',
    paragraph_delimiter='\n\n',
):
    """Get the lengths of the paragraphs in each file in the corpus.

    Args:
        corpus (str): Path to the corpus directory.
        text_key (str): Key of the text in the JSON file. Default is 'text'.
        You might want to change to 'text_deduped'.
        paragraph_delimiter (str): Delimiter that separates paragraphs in the
            text. Default is '\n\n'.

    Returns:
        list: A list of lists. Each inner list contains the lengths of the
            paragraphs in a file.
    """

    parlens = []

    for file in os.listdir(corpus):
        file_paras = []
        with open(os.path.join(corpus, file), 'r') as f:
            data = json.load(f)
        text = data[text_key]
        paragraphs = text.split(paragraph_delimiter)
        for paragraph in paragraphs:
            file_paras.append(len(paragraph))
        parlens.append(file_paras)

    return parlens


def plot_avg_parlens(
    corpus,
    text_key='text',
    paragraph_delimiter='\n\n',
    bins=100,
):
    """Plot the distribution of average paragraph lengths in the corpus.

    Args:
        corpus (str): Path to the corpus directory.
        text_key (str): Key of the text in the JSON file. Default is 'text'.
            You might want to change to 'text_deduped'.
        paragraph_delimiter (str): Delimiter that separates paragraphs in the
            text. Default is '\n\n'.
        bins (int): Number of bins for the histogram.
    """

    parlens = get_parlens(
        corpus, text_key, paragraph_delimiter
    )

    avg_lens = [np.mean(paragraphs) for paragraphs in parlens]

    plt.hist(avg_lens, bins=bins)
    plt.title('Average Paragraph Lengths')
    plt.xlabel('Average Paragraph Length')
    plt.ylabel('Frequency')

    plt.show()

    return


def print_parlen_threshold(
    corpus,
    threshold,
    below_threshold=True,
    text_key='text',
):
    """Print texts with average paragraph lengths above or below a certain
    threshold.

    Args:
        corpus (str): Path to the corpus directory.
        threshold (int): The threshold value. (Texts with exactly this length
            will not be printed.)
        below_threshold (bool): If True, texts below the threshold will be
            printed, else texts above the threshold will be printed.
        text_key (str): Key of the text in the JSON file. Default is 'text'.
            You might want to change to 'text_deduped'.
    """

    parlens = get_parlens(corpus, text_key)
    filelist = os.listdir(corpus)

    criterion = threshold_checker_factory(
        threshold, below_threshold
    )

    for i, file_paras in enumerate(parlens):
        avg_len = np.mean(file_paras)
        if criterion(avg_len):
            with open(os.path.join(corpus, filelist[i]), 'r') as f:
                data = json.load(f)
            print(filelist[i].upper())
            print()
            print(data[text_key])
            print()
            print(f'Average Paragraph Length: {avg_len}')
            print()
            print('+'*30)
            print()

    return


def remove_avg_parlen_threshold(
    corpus,
    threshold,
    below_threshold=True,
    text_key='text',
    force=False,
    title_key=None,
):
    """Remove texts with average paragraph lengths above or below a certain
    threshold.

    Args:
        corpus (str): Path to the corpus directory.
        threshold (int): The threshold value. (Texts with exactly this length
            will not be removed.)
        below_threshold (bool): If True, texts below the threshold will be
            removed, else texts above the threshold will be removed.
        text_key (str): Key of the text in the JSON file. Default is 'text'.
            You might want to change to 'text_deduped'.
        force (bool): If True, the function will remove the files without
            asking for confirmation. Default is False.
        title_key (str): If not None, the function will print the value of this
            key in the data when asking for confirmation. Default is None.
    """

    parlens = get_parlens(corpus, text_key)
    filelist = os.listdir(corpus)

    criterion = threshold_checker_factory(
        threshold, below_threshold
    )

    for i, file_paras in enumerate(parlens):
        avg_len = np.mean(file_paras)
        if criterion(avg_len):
            filename = filelist[i]
            delete_with_confirmation(
                corpus,
                filename,
                force,
                title_key,
            )

    return


# CHECK FOR UNWANTED LANGUAGES

def detect_languages(
    corpus,
    text_key='text',
):
    """Detect the languages of the texts in the corpus.
    Use langdetect to do so.

    Args:
        corpus (str): Path to the corpus directory.
        text_key (str): Key of the text in the JSON file. Default is 'text'.
        You might want to change to 'text_deduped'.

    Returns:
        list: A list of strs of the detected languages.
    """

    filelist = utils.listdir_filetype(corpus, '.json', absolute=False)
    languages = []

    for file in tqdm(filelist):
        try:
            with open(os.path.join(corpus, file), 'r') as f:
                data = json.load(f)
            text = data[text_key]
            language = detect(text)
            languages.append(language)
        except LangDetectException:
            languages.append('Unknown')

    return languages


def print_languages(
    corpus,
    relevant_languages,
    precompiled_languages=None,
    text_key='text',
):
    """Print texts in the languages you are interested in.

    Args:
        corpus (str): Path to the corpus directory.
        relevant_languages (list): List of languages you are interested in.
        precompiled_languages (list): List of dectected languages. Should have
            same length as the number of files in the corpus. If None, it
            will first detect the languages of the texts itself.
            The reason for this argument is that the language detection can
            take a long time, so you might want to save the detected
            languages and pass them to the function. (All functions in this
            module will return a list you can pass here.)
        text_key (str): Key of the text in the JSON file. Default is 'text'.
        You might want to change to 'text_deduped'.

    Returns:
        list: A list of strs of the detected languages.
    """

    if isinstance(precompiled_languages, list):
        languages = precompiled_languages
    else:
        languages = detect_languages(corpus, text_key)
    filelist = os.listdir(corpus)

    for i, language in enumerate(languages):
        if language in relevant_languages:
            with open(os.path.join(corpus, filelist[i]), 'r') as f:
                data = json.load(f)
            text = data[text_key]

            print(filelist[i].upper())
            print()
            print(text)
            print()
            print(f'Detected language: {language}')
            print()
            print('+'*30)
            print()

    return languages


def remove_languages(
    corpus,
    languages_to_remove,
    precompiled_languages=None,
    text_key='text',
    force=False,
    title_key=None,
):
    """Remove texts in unwanted languages.

    Args:
        corpus (str): Path to the corpus directory.
        languages_to_remove (list): List of languages you want to remove.
        precompiled_languages (list): List of dectected languages. Should have
            same length as the number of files in the corpus. If None, it
            will first detect the languages of the texts itself.
            The reason for this argument is that the language detection can
            take a long time, so you might want to save the detected
            languages and pass them to the function. (All functions in this
            module will return a list you can pass here.)
        text_key (str): Key of the text in the JSON file. Default is 'text'.
            You might want to change to 'text_deduped'.
        force (bool): If True, the function will remove the files without
            asking for confirmation. Default is False.
        title_key (str): If not None, the function will print the value of this
            key in the data when asking for confirmation. Default is None.
    """

    if isinstance(precompiled_languages, list):
        languages = precompiled_languages
    else:
        languages = detect_languages(corpus, text_key)
    filelist = os.listdir(corpus)

    for i, lang in enumerate(languages):
        if lang in languages_to_remove:
            delete_with_confirmation(
                corpus,
                filelist[i],
                force,
                title_key,
            )

    return


def plot_languages(
    corpus,
    precompiled_languages=None,
    text_key='text',
):
    """Plot the distribution of detected languages in the corpus.

    Args:
        corpus (str): Path to the corpus directory.
        precompiled_languages (list): List of dectected languages. Should have
            same length as the number of files in the corpus. If None, it
            will first detect the languages of the texts itself.
            The reason for this argument is that the language detection can
            take a long time, so you might want to save the detected
            languages and pass them to the function. (All functions in this
            module will return a list you can pass here.)
        text_key (str): Key of the text in the JSON file. Default is 'text'.
            You might want to change to 'text_deduped'.

    Returns:
        list: A list of strs of the detected languages.
    """

    if isinstance(precompiled_languages, list):
        languages = precompiled_languages
    else:
        languages = detect_languages(corpus, text_key)

    plt.hist(languages)
    plt.title('Detected Languages')
    plt.xlabel('Language')
    plt.ylabel('Frequency')

    plt.show()

    return languages


# CUSTOM REMOVAL


def print_custom_removal(
    corpus,
    custom_checker,
    key=None
):
    """Print texts that should be removed according to a custom checker.

    Args:
        corpus (str): Path to the corpus directory.
        custom_checker (function): A function that takes the data of a
            json file (i.e. a dict) as input and returns False if the file
            should be removed and True if it should be kept.
        key (str): If not None, the function will print the value of this key
            in the data. Default is None.
    """

    filelist = os.listdir(corpus)

    for i, file in enumerate(filelist):
        with open(os.path.join(corpus, file), 'r') as f:
            data = json.load(f)
        if not custom_checker(data):
            print(file.upper())
            print()
            if key is not None:
                print(data[key])
            else:
                print(data)
            print()
            print('+'*30)
            print()

    return


def apply_custom_removal(
    corpus,
    custom_checker,
    force=False,
    title_key=None,
):
    """Apply a custom removal function to the corpus.

    Args:
        corpus (str): Path to the corpus directory.
        custom_checker (function): A function that takes the data of a
            json file (i.e. a dict) as input and returns False if the file
            should be removed and True if it should be kept.
        force (bool): If True, the function will remove the files without
            asking for confirmation. Default is False.
        title_key (str): If not None, the function will print the value of this
            key in the data when asking for confirmation. Default is None.
    """

    filelist = os.listdir(corpus)

    for i, file in enumerate(filelist):
        with open(os.path.join(corpus, file), 'r') as f:
            data = json.load(f)
        if not custom_checker(data):
            delete_with_confirmation(
                corpus,
                file,
                force,
                title_key,
            )
    return


# NLP FOR APPLICATION

def add_lemma_token(
    corpus,
    text_key='text_deduped',
    spacy_model='de_core_news_lg',
):
    """Add keys with lemmas and tokens to the JSON files in the corpus.

    Args:
        corpus (str): Path to the corpus directory.
        text_key (str): Key of the text in the JSON file. Default is
            'text_deduped'.
        spacy_model (str): Name of the spaCy model to use. Default is
            'de_core_news_lg'.
    """

    nlp = spacy.load(spacy_model)

    for f, data in tqdm(
        utils.jsondir_filepath_data_gen(corpus, absolute=True),
        total=len(utils.listdir_filetype(corpus, '.json'))
    ):
        text = data[text_key]
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc]
        lexems = [token.text for token in doc]
        data['lemmas'] = lemmas
        data['token'] = lexems

        utils.write_json(data, f, overwrite=True)

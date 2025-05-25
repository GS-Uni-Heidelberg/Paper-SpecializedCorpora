import re
import nltk


def contains_alphab(string: str, lang: str = 'de'):
    if lang == 'de':
        if re.search(r'[a-zA-ZäÄÜüÖöß]', string):
            return True
    elif lang == 'en':
        if re.search(r'[a-zA-Z]', string):
            return True
    else:
        raise ValueError(f"Language '{lang}' is not implemented.")
    return None


def contains_alphab_tuple(tuple_: tuple, lang: str = 'de'):
    return all(contains_alphab(string, lang) for string in tuple_)


def tuple_to_regex(word_tuple):
    regex = r'\b' + r'\b.*?\b'.join(map(re.escape, word_tuple)) + r'\b'
    return regex


def begin_end_stopword(tuple, language='de'):
    """Check for stopwords at the beginning or end of a tuple."""
    return is_stopword(tuple[0], language) or is_stopword(tuple[-1], language)


def is_stopword(word, language='de'):
    if language == 'de':
        stopwords = nltk.corpus.stopwords.words('german')
    elif language == 'en':
        stopwords = nltk.corpus.stopwords.words('english')
    else:
        raise ValueError(f"Language '{language}' is not implemented.")
    stopwords = set(stopwords.lower() for stopwords in stopwords)
    return word.lower() in stopwords

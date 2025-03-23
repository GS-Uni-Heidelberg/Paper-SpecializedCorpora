from collections import Counter
from . import token_util as utils
from typing import Callable


class Corpus:
    """
    A class to represent a corpus of lemmatized documents.

    Attributes
    ----------
    documents : list[list[str]]
        A list of lemmatized documents.
    filter : None | callable
        A function to filter out unwanted words,
        taking a word and a language as arguments,
        and returning True if the word is to be kept.
        (E.g. stopwords, punctuation, etc.)
        Default is utils.contains_alphab (A function that
        checks if a token contains an alphabet character)
    language : str
        The language of the documents.
    """

    def __init__(
        self,
        documents: list[list[str]],
        filter: None | Callable = utils.contains_alphab,
        language: str = 'de'
    ):
        self.filter = filter
        self._language = language
        self.documents = documents

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, filter):
        if filter is None:
            self._filter = lambda x, y: True
        elif callable(filter):
            self._filter = filter
        else:
            return TypeError("Filter must be a callable or None")

    @property
    def documents(self):
        return self._documents

    @documents.setter
    def documents(self, documents):
        clean_documents = []
        for doc in documents:
            clean_documents.append(
                [
                    word for word in doc
                    if self.filter(word, self._language)
                ]
            )
        self._documents = clean_documents

    def treat_as_one(self, ngram, name=None):
        """Function to treat an ngram as a single token
        in the entire corpus."""
        ngram = list(ngram)

        if name is None:
            name = ' '.join(ngram)
        for i, doc in enumerate(self.documents):
            new_doc = []
            j = 0
            while j < len(doc):
                if j <= len(doc) - len(ngram) and doc[j:j+len(ngram)] == ngram:
                    new_doc.append(name)
                    j += len(ngram)
                else:
                    new_doc.append(doc[j])
                    j += 1
            self.documents[i] = new_doc


class FrequencyCorpus(Corpus):

    def __init__(
        self,
        docs: list[list[str]],
        filter: None | Callable = utils.contains_alphab,
        language: str = 'de'
    ):
        super.__init__(docs, filter, language)
        self.size = dict()
        self.unique = dict()
        self.ngram_counts = dict()
        self.ngram_doccounts = dict()

    def get_ngrams(self, n):
        ngram_counts = self.ngram_counts.get(n, None)

        if ngram_counts is not None:
            return ngram_counts

        ngrams = []
        ngram_doccount = {}

        for doc in self.documents:
            seen_ngrams = set()
            for i in range(len(doc) - n + 1):
                ngram = tuple(doc[i:i + n])
                if all(self.filter(word, self.language) for word in ngram):
                    ngrams.append(ngram)
                    if ngram not in seen_ngrams:
                        ngram_doccount[ngram] = (
                            ngram_doccount.get(ngram, 0)
                            + 1
                        )
                        seen_ngrams.add(ngram)

        self.ngram_doccounts[n] = ngram_doccount
        self.ngram_counts[n] = dict(Counter(ngrams))
        self.unique[n] = len(self.ngram_counts[n])
        self.size[n] = len(ngrams)

        return self.ngram_counts[n]

    def get_unigrams(self):
        return self.get_ngrams(1)

    def get_bigrams(self):
        return self.get_ngrams(2)

    def get_trigrams(self):
        return self.get_ngrams(3)

import re
from tqdm import tqdm
from .. import token_util as utils
from ..corpus import Corpus


class term_counts:
    @property
    def term(self):
        return self._term

    # term.setter
    @term.setter
    def term(self, value):
        self._term = value

    def __init__(self, term, base_term, term_count, coocurrence_count):
        self.term = term
        self.base_term = base_term
        self.term_count = term_count
        self.coocurrence_count = coocurrence_count

    def qtr(self):
        if self.term_count == 0:
            raise ZeroDivisionError("term_count cannot be 0")
        return self.coocurrence_count / self.term_count

    def rqtr(self, qtr_base):
        qtr = self.qtr()
        if qtr_base == 0:
            raise ZeroDivisionError("qtr_base cannot be 0")
        return 100 * (qtr - qtr_base) / qtr_base

    def rqtrn(self, qtr_base):
        if qtr_base == 1:
            raise ZeroDivisionError("qtr_base cannot be 1")
        qtr = self.qtr()
        return 100 * (qtr - qtr_base) / (1 - qtr_base)

    def __str__(self):
        return (
            f"{self.term}: {self.term_count}, "
            f"mit <{self.base_term}>: {self.coocurrence_count}"
        )


# def get_values(
#     terms: list[str],
#     core_term: str,
#     qtr_base: float,
#     corpus: Corpus,
#     lower=True,
# ):

#     counts = []
#     for term in terms:
#         try:
#             counts.append(term_counts(term, core_term, 0, 0))
#         except re.error:
#             counts.append(None)

#     for doc in tqdm(corpus.lemmas):
#         doc = set(doc)
#         for term in counts:
#             if term is None:
#                 continue
#             contains_core_term = (core_term[0] in doc or core_term[1] in doc)
#             if term.term in doc:
#                 term.term_count += 1
#                 if contains_core_term:
#                     term.coocurrence_count += 1

#     return counts


# def get_possible_values(
#     core_terms, qtr_base, corpus, lower=True,
#     min_count=3
# ):

#     relevant_docs = []
#     for doc in tqdm(
#         corpus, desc="Finding relevant documents with the core term"
#     ):
#         doccopy = set(doc)
#         if core_terms[1] in doc or core_terms[2] in doccopy:
#             relevant_docs.append(doc)

#     all_words = all_corpus_words(relevant_docs)
#     all_words = [word for word in all_words if all_words[word] >= min_count]

#     print(f'Found {len(all_words)} relevant words in the corpus')

#     values = get_values(
#         all_words, core_terms, qtr_base, corpus, lower
#     )

#     return values


def qtr_baseline(core_term_1, core_term_2, corpus, verbose=True):
    counts1 = term_counts(core_term_1, core_term_2, 0, 0)
    counts2 = term_counts(core_term_2, core_term_1, 0, 0)

    corpus = corpus.documents

    for doc in corpus:
        doc = set(doc)

        found1 = False
        found2 = False
        if core_term_1 in doc:
            counts1.term_count += 1
            found1 = True
        if core_term_2 in doc:
            counts2.term_count += 1
            found2 = True
        if found1 and found2:
            counts1.coocurrence_count += 1
            counts2.coocurrence_count += 1

    if counts1.term_count == 0:
        raise ValueError(f"Term '{core_term_1}' not found in corpus")
    if counts2.term_count == 0:
        raise ValueError(f"Term '{core_term_2}' not found in corpus")

    qtr1 = counts1.qtr()
    qtr2 = counts2.qtr()

    if qtr1 == qtr2:
        print("Both terms are equally good, returning the first term")
        lower_qtr = qtr1
        better_term = core_term_1
    elif qtr1 > qtr2:
        lower_qtr = qtr2
        better_term = core_term_2
    else:
        lower_qtr = qtr1
        better_term = core_term_1

    if verbose:
        print(f"Term {counts1.term}: {counts1.term_count}, QTR: {qtr1}")
        print(f"Term {counts2.term}: {counts2.term_count}, QTR: {qtr2}")
        print(f"Both terms coocurring: {counts1.coocurrence_count}")
        print()
        print(f'Baseline term (with lower QTR): {better_term}')

    return lower_qtr, better_term


# def all_corpus_words(
#     corpus
# ):
#     all_words = {}

#     for doc in corpus:
#         for word in doc:
#             all_words[word] = all_words.get(word, 0) + 1

#     return all_words


def all_corpus_ngrams(
    corpus, n=2
):
    all_words = {}

    for doc in corpus:
        doclen = len(doc)
        for i, _ in enumerate(doc):
            if i == doclen - n + 1:
                break
            ngram = tuple(doc[i:i+n])
            all_words[ngram] = all_words.get(ngram, 0) + 1
    return all_words


def get_all_ngrams(
    core_terms,
    qtr_base,
    corpus,
    min_count=3,
    n=1
):

    relevant_docs = []
    for doc in tqdm(
        corpus, desc="Finding relevant documents with the core term"
    ):
        doccopy = set(doc)
        if core_terms[0] in doc or core_terms[1] in doccopy:
            relevant_docs.append(doc)

    all_words = all_corpus_ngrams(relevant_docs, n=n)
    cleaned_all_words = []
    for word in all_words:
        if all_words[word] >= min_count:
            if not utils.begin_end_stopword(word, 'de'):
                if (
                    core_terms[0] not in word
                    and core_terms[1] not in word
                    and '--' not in word
                    and '\n\n' not in word
                ):
                    cleaned_all_words.append(word)

    print(
        f'Found {len(cleaned_all_words)} words in the corpus'
        f' coocurring with {core_terms[0]} and {core_terms[1]}'
    )

    values = get_ngram_values(
        cleaned_all_words, core_terms, qtr_base, corpus
    )

    return values


def get_ngram_values(
    ngram_list, core_terms, qtr_base, corpus
):

    n = len(ngram_list[0])

    counts = []
    for term in ngram_list:
        counts.append(term_counts(term, core_terms, 0, 0))

    for doc in tqdm(corpus):
        doclen = len(doc)
        doc_tuples = set()
        contains_core_term = (core_terms[0] in doc or core_terms[1] in doc)

        for i, word in enumerate(doc):
            if i == doclen - n + 1:
                break
            ngram = tuple(doc[i:i+n])
            doc_tuples.add(ngram)

        for term in counts:
            if term.term in doc_tuples:
                term.term_count += 1
                if contains_core_term:
                    term.coocurrence_count += 1

    return counts

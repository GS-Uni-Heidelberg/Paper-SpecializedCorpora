from tqdm import tqdm
from .. import token_util as utils
import pandas as pd


class term_counts:
    @property
    def term(self):
        return self._term

    # term.setter
    @term.setter
    def term(self, value):
        self._term = value

    def __init__(self, term, base_term, term_count, cooccurence_count):
        self.term = term
        self.base_term = base_term
        self.term_count = term_count
        self.cooccurence_count = cooccurence_count

    def qtr(self):
        if self.term_count == 0:
            raise ZeroDivisionError("term_count cannot be 0")
        return self.cooccurence_count / self.term_count

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
            f"mit <{self.base_term}>: {self.cooccurence_count}"
        )


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
            counts1.cooccurence_count += 1
            counts2.cooccurence_count += 1

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
        print(f"Both terms coocurring: {counts1.cooccurence_count}")
        print()
        print(f'Baseline term (with lower QTR): {better_term}')

    return lower_qtr, better_term


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


def count_cooccurence_ngram(
    core_terms,
    corpus,
    min_count=3,
    n=1
):

    corpus = corpus.documents

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
                    and '--' not in word  # skip punctuation
                    and '\n\n' not in word  # skip paragraph breaks
                ):
                    cleaned_all_words.append(word)

    print(
        f'Found {len(cleaned_all_words)} words in the corpus'
        f' coocurring with {core_terms[0]} and {core_terms[1]}'
    )

    values = get_ngram_values(
        cleaned_all_words, core_terms, corpus
    )

    return values


def count_cooccurence(
    core_terms,
    corpus,
    min_count=3,
    max_ngram_len=3,
):
    result_list = []
    for n in range(1, max_ngram_len + 1):
        result_list.extend(
            count_cooccurence_ngram(
                core_terms, corpus, min_count=min_count, n=n
            )
        )

    return result_list


def get_ngram_values(
    ngram_list, core_terms, corpus
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
                    term.cooccurence_count += 1

    return counts


def cooccurence_to_metric(
    cooccurence_values,
    baseline,
    metric='rqtrn',
):

    match metric.lower():
        case 'qtr':
            df = pd.DataFrame(
                [(term.term, term.qtr()) for term in cooccurence_values],
                columns=['Term', 'QTR']
            )
        case 'rqtr':
            df = pd.DataFrame(
                [
                    (term.term, term.rqtr(baseline))
                    for term in cooccurence_values
                ],
                columns=['Term', 'RQTR']
            )
        case 'rqtrn':
            df = pd.DataFrame(
                [
                    (term.term, term.rqtrn(baseline))
                    for term in cooccurence_values
                ],
                columns=['Term', 'RQTRN']
            )
        case _:
            raise ValueError(
                f"Unknown metric: {metric}. "
                f"Please use 'qtr', 'rqtr', or 'rqtrn'."
            )

    df = df.sort_values(by=metric.upper(), ascending=False)
    df = df.reset_index(drop=True)

    return df

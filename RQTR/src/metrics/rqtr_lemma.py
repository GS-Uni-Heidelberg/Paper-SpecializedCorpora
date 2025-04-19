from tqdm import tqdm
from .. import token_util as utils
import pandas as pd
from ..corpus import Corpus
from typing import Iterable, Callable
from collections import defaultdict


class TermCounts:
    """Class to represent term frequencies and term co-occurrences
    in documents.
    Includes methods to calculate QTR, RQTR, and RQTRN metrics.

    Attributes:
        term (str or tuple[str]): The term or n-gram being counted.
        base_terms (Iterable[str]): The base terms for co-occurrence
            counting.
        term_count (int): The number of documents containing the term.
        cooccurence_count (int): The number of documents containing both
            the term and the base terms.
    """

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        self._term = value

    def __init__(
        self,
        term: str | tuple[str],
        base_term: Iterable[str],
        term_count: int,
        cooccurence_count: int
    ):
        self.term = term
        self.base_term = base_term
        self.term_count = term_count
        self.cooccurence_count = cooccurence_count

    def qtr(self):
        """Calculate QTR (Query Term Relevance)"""
        if self.term_count == 0:
            raise ZeroDivisionError("term_count cannot be 0")
        return self.cooccurence_count / self.term_count

    def rqtr(self, qtr_base: float):
        """Calculate RQTR (Relative Query Term Relevance)"""
        qtr = self.qtr()
        if qtr_base == 0:
            raise ZeroDivisionError("qtr_base cannot be 0")
        return 100 * (qtr - qtr_base) / qtr_base

    def rqtrn(self, qtr_base: float):
        """Calculate RQTRN (Normalized Relative Query Term Relevance)"""
        if qtr_base == 1:
            raise ZeroDivisionError("qtr_base cannot be 1")
        qtr = self.qtr()
        return 100 * (qtr - qtr_base) / (1 - qtr_base)

    def __str__(self):
        return (
            f"{self.term}: {self.term_count}, "
            f"mit <{self.base_term}>: {self.cooccurence_count}"
        )

    def __repr__(self):
        return (
            f"TermCounts(term={self.term}, "
            f"base_terms={self.base_term}, "
            f"term_count={self.term_count}, "
            f"cooccurence_count={self.cooccurence_count})"
        )


class SearchTerms:
    @property
    def terms(self):
        return self._terms

    @terms.setter
    def terms(self, value):
        # Validate terms
        if len(value) != len(set(value)):
            raise ValueError("Duplicate terms found")
        if not all(isinstance(term, (str, tuple)) for term in value):
            raise TypeError("Terms must be str or tuple[str]")

        self._terms = {
            term if isinstance(term, tuple) else (term,)
            for term in value
        }

        # Update term lens - more efficient approach
        self._term_lens = defaultdict(set)
        for term in value:
            term_len = len(term) if isinstance(term, tuple) else 1
            self._term_lens[term_len].add(term)
        self._term_lens = dict(self._term_lens)

    def __init__(self, terms: Iterable[str | tuple[str]]):
        self._term_lens = {}
        self.terms = terms

    def in_doc(
        self,
        doc: list[str],
    ) -> set[str | tuple[str]]:
        # For each term len, create a set out of the doc
        # and check if any of the terms are in the doc

        terms_in_doc = set()

        for term_len in self._term_lens:
            if term_len == 1:
                # Check for single terms
                docset = set(doc)
                docset = {
                    (term,) if isinstance(term, str) else term
                    for term in docset
                }
                for term in self._term_lens[term_len]:
                    if term in docset:
                        terms_in_doc.add(term)
            else:
                # Check for n-grams
                doc_tuples = set()
                doclen = len(doc)
                for i, _ in enumerate(doc):
                    if i == doclen - term_len + 1:
                        break
                    ngram = tuple(doc[i:i+term_len])
                    doc_tuples.add(ngram)
                for term in self._term_lens[term_len]:
                    if term in doc_tuples:
                        terms_in_doc.add(term)

        return terms_in_doc


def qtr_baseline(
    core_terms: list[str | tuple],
    corpus: Corpus,
    verbose=True
):

    if len(core_terms) < 2:
        raise ValueError("At least 2 core terms are required")

    # Initialize counts for each term
    all_counts = {}
    for term in core_terms:
        all_counts[term] = TermCounts(
            term,
            core_terms,
            0,
            0
        )

    search_terms = SearchTerms(core_terms)

    # Count occurrences and co-occurrences
    for doc, _ in corpus:

        # Track which terms are found in this document
        found_terms = search_terms.in_doc(doc)

        # Update co-occurrence counts
        seen_cooccurrences = set()
        for term1 in found_terms:
            all_counts[term1].term_count += 1
            for term2 in found_terms:
                if term1 == term2:
                    continue
                if term1 in seen_cooccurrences:
                    continue
                all_counts[term1].cooccurence_count += 1
                seen_cooccurrences.add(term1)
                break

    # Check if any terms are missing from corpus
    for term, counts in all_counts.items():
        if counts.term_count == 0:
            raise ValueError(f"Term '{term}' not found in corpus")

    # Calculate QTR for each term
    qtr_values = {term: counts.qtr() for term, counts in all_counts.items()}

    # Find term with lowest QTR
    better_term = min(qtr_values.items(), key=lambda x: x[1])
    lower_qtr = better_term[1]
    better_term = better_term[0]

    if verbose:
        for term, counts in all_counts.items():
            print(f"Term {term}: {counts.term_count}, QTR: {qtr_values[term]}")

        print("\nCo-occurrence counts:")
        for term1 in core_terms:
            for term2 in core_terms:
                if term1 != term2:
                    print(
                        f"Co-occurrence ({term1}, {term2}): "
                        f"{all_counts[term1].cooccurence_count}"
                    )
        print("\nQTR values:")
        for term, qtr in qtr_values.items():
            print(f"QTR({term}): {qtr}")

        print(f"\nBaseline term (with lowest QTR): {better_term}")

    return lower_qtr, better_term


def all_corpus_ngrams(
    docs, n=2
):
    all_words = set()

    for doc in docs:

        doclen = len(doc)
        for i, _ in enumerate(doc):
            if i == doclen - n + 1:
                break
            ngram = tuple(doc[i:i+n])
            all_words.add(ngram)
    return all_words


def count_cooccurence_ngram(
    core_terms: list[str | tuple],
    corpus: Corpus,
    n: int = 1,
    filter_func: Callable = utils.begin_end_stopword
):

    core_terms = set(core_terms)
    core_search_terms = SearchTerms(core_terms)

    relevant_docs = []
    for doc, _ in tqdm(
        corpus, desc="Finding relevant documents with the core term"
    ):
        if len(core_search_terms.in_doc(doc)) > 0:
            relevant_docs.append(doc)

    all_words = all_corpus_ngrams(relevant_docs, n=n)
    all_words_cleaned = {}
    for word in all_words:
        if word in core_terms:
            continue
        if word == '\n\n':
            continue
        if word == '--':
            continue
        if filter_func(word):
            continue
        all_words_cleaned[word] = TermCounts(
            word,
            core_terms,
            0,
            0
        )

    full_search = set(all_words_cleaned.keys()) | core_terms
    search_terms = SearchTerms(full_search)

    print(
        f'Found {len(all_words_cleaned)} words in the corpus'
        f' coocurring with {core_terms}'
    )

    for doc, _ in corpus:
        search_result = search_terms.in_doc(doc)
        intersect_core = core_terms.intersection(search_result)
        intersect_canditates = search_result - core_terms

        for term in intersect_canditates:
            all_words_cleaned[term].term_count += 1
            if len(intersect_core) > 0:
                all_words_cleaned[term].cooccurence_count += 1

    return list(all_words_cleaned.values())


def count_cooccurence(
    core_terms,
    corpus,
    min_count=1,
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

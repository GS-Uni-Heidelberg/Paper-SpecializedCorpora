import re
from tqdm import tqdm
import utils


class term_counts:
    @property
    def term(self):
        return self._term

    # term.setter
    @term.setter
    def term(self, value):
        try:
            re.compile(value)
            self._term = value
        except re.error:
            raise re.error("Term must be a valid regular expression")

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


def get_values(
    terms, core_term, qtr_base, corpus, lower=True,
    split_first=False,
    split_pattern=r'\b'
):
    if lower:
        corpus = [doc.lower() for doc in corpus]

    counts = []
    for term in terms:
        try:
            counts.append(term_counts(term, core_term, 0, 0))
        except re.error:
            counts.append(None)

    for doc in tqdm(corpus):
        if split_first:
            doc = re.split(split_pattern, doc)
            doc = set(doc)
            for term in counts:
                if term.term in doc:
                    term.term_count += 1
                    if core_term[1] in doc or core_term[0] in doc:
                        term.coocurrence_count += 1
        else:
            for term in counts:
                if term is None:
                    continue
                contains_core_term = any((
                    re.search(core_term[0], doc),
                    re.search(core_term[1], doc)
                ))
                if re.search(term.term, doc):
                    term.term_count += 1
                    if contains_core_term:
                        term.coocurrence_count += 1

    return counts


def get_possible_values(
    core_terms, qtr_base, corpus, lower=True,
    min_count=3
):
    if lower:
        corpus = [doc.lower() for doc in corpus]

    relevant_docs = []
    for doc in tqdm(
        corpus, desc="Finding relevant documents with the core term"
    ):
        if re.search(f'{core_terms[0]}|{core_terms[1]}', doc):
            relevant_docs.append(doc)

    all_words = all_corpus_words(relevant_docs, try_compile=True)
    all_words = [word for word in all_words if all_words[word] >= min_count]

    print(f'Found {len(all_words)} relevant words in the corpus')

    values = get_values(
        all_words, core_terms, qtr_base, corpus, lower, split_first=True
    )

    return values


def qtr_baseline(core_term_1, core_term_2, corpus, verbose=True, lower=True):
    counts1 = term_counts(core_term_1, core_term_2, 0, 0)
    counts2 = term_counts(core_term_2, core_term_1, 0, 0)

    for doc in corpus:
        if lower:
            doc = doc.lower()
        found1 = False
        found2 = False
        if re.search(core_term_1, doc):
            counts1.term_count += 1
            found1 = True
        if re.search(core_term_2, doc):
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
        print(f"Both terms: {counts1.coocurrence_count}")

    return lower_qtr, better_term


def all_corpus_words(
    corpus, return_format=dict,
    tokenization_pattern=None, lower=True, try_compile=False,
):
    all_words = {}
    if tokenization_pattern is None:
        # Split at and exclude these chars:
        tokenization_pattern = r"\b"

    for doc in corpus:
        if lower:
            doc = doc.lower()
        words = re.split(tokenization_pattern, doc)
        for word in words:
            all_words[word] = all_words.get(word, 0) + 1

    all_words = {
        word: count for word, count in all_words.items()
        if utils.contains_alphab(word)
    }

    if try_compile:
        compilable_words = {}
        for word, count in all_words.items():
            try:
                re.compile(word)
                compilable_words[word] = count
            except re.error:
                print(f"Word '{word}' is not a valid regular expression")
        all_words = compilable_words

    if return_format == dict:
        pass
    elif return_format == list:
        all_words = list(all_words.keys())
    elif return_format == set:
        all_words = set(all_words.keys())

    return all_words


def all_corpus_ngrams(
    corpus, n=2, return_format=dict,
    tokenization_pattern=None, lower=True, try_compile=False,
):
    all_words = {}
    if tokenization_pattern is None:
        # Split at and exclude these chars:
        tokenization_pattern = r"\b"

    for doc in corpus:
        if lower:
            doc = doc.lower()
        words = re.split(tokenization_pattern, doc)

        words = [word for word in words if utils.contains_alphab(word)]

        doclen = len(words)
        for i, word in enumerate(words):
            if i == doclen - n + 1:
                break
            ngram = tuple(words[i:i+n])
            all_words[ngram] = all_words.get(ngram, 0) + 1

    all_words = {
        word: count for word, count in all_words.items()
        if utils.contains_alphab_tuple(word)
    }

    if try_compile:
        compilable_words = {}
        for word, count in all_words.items():
            try:
                for w in word:
                    re.compile(w)
                compilable_words[word] = count
            except re.error:
                print(f"Word '{word}' is not a valid regular expression")
        all_words = compilable_words

    if return_format == dict:
        pass
    elif return_format == list:
        all_words = list(all_words.keys())
    elif return_format == set:
        all_words = set(all_words.keys())

    return all_words


def get_possible_ngrams(
    core_terms, qtr_base, corpus, lower=True,
    min_count=3
):
    if lower:
        corpus = [doc.lower() for doc in corpus]

    relevant_docs = []
    for doc in tqdm(
        corpus, desc="Finding relevant documents with the core term"
    ):
        if re.search(f'{core_terms[0]}|{core_terms[1]}', doc):
            relevant_docs.append(doc)

    all_words = all_corpus_ngrams(relevant_docs, try_compile=True)
    cleaned_all_words = []
    for word in all_words:
        if all_words[word] >= min_count:
            if not utils.begin_end_stopword(word, 'de'):
                if core_terms[0] not in word and core_terms[1] not in word:
                    cleaned_all_words.append(word)
    all_words = [
        utils.tuple_to_regex(word) for word in cleaned_all_words
    ]

    print(f'Found {len(all_words)} relevant words in the corpus')

    values = get_values(
        all_words, core_terms, qtr_base, corpus, lower, split_first=False
    )

    return values

import re
from ..corpus import Corpus, FrequencyCorpus


def surround_with_undsc(
    text: str,
) -> str:
    return f"___{text}___"


def wordlist_edit(
    wordlist
):
    new_wordlist = []
    for ngram in wordlist:
        if isinstance(ngram, str):
            new_wordlist.append(
                surround_with_undsc(ngram)
            )
        elif isinstance(ngram, tuple) and len(ngram) == 1:
            new_wordlist.append(
                surround_with_undsc(ngram[0])
            )
        elif isinstance(ngram, tuple) and len(ngram) > 1:
            new_wordlist.append(
                surround_with_undsc('___'.join(ngram))
            )
        else:
            raise ValueError(
                "Invalid wordlist format."
                "Entries must be strings or tuples."
            )
    return new_wordlist


def worddict_edit(
    worddict
):
    new_worddict = {}
    for ngram, weight in worddict.items():
        if isinstance(ngram, str):
            new_worddict[surround_with_undsc(ngram)] = weight
        elif isinstance(ngram, tuple) and len(ngram) == 1:
            new_worddict[surround_with_undsc(ngram[0])] = weight
        elif isinstance(ngram, tuple) and len(ngram) > 1:
            new_worddict[surround_with_undsc('___'.join(ngram))] = weight
        else:
            raise ValueError(
                "Invalid wordlist format."
                "Entries must be strings or tuples."
            )
    return new_worddict


def match_wordlist(
    corpus: Corpus,
    wordlist: list | set,
    min: int = 1,
    unique: bool = False,
):
    """Function to treat a list of words as a single token
    in the entire corpus."""
    words = wordlist_edit(wordlist)
    escaped_words = [re.escape(word) for word in words]
    pattern = re.compile(f"({'|'.join(escaped_words)})")

    found_docs_id = []
    for i, entry in enumerate(corpus):
        doc, metadata = entry
        regex_doc = surround_with_undsc('___'.join(doc))
        if unique:
            found_words = set()
            for match in pattern.finditer(regex_doc):
                found_words.add(match.group())
            if len(found_words) >= min:
                found_docs_id.append(i)
        else:
            if len(pattern.findall(regex_doc)) >= min:
                found_docs_id.append(i)

    return found_docs_id


def match_weighted_wordlist(
    corpus: Corpus,
    wordlist: dict,
    min: int = 1,
    unique: bool = False,
):
    """Function to treat a weighted list of words as a single token
    in the entire corpus. Returns documents where the sum of weights
    of matched words is at least equal to min."""

    words = worddict_edit(wordlist)
    escaped_words = [re.escape(word) for word in words]
    pattern = re.compile(f"({'|'.join(escaped_words)})")

    found_docs_id = []
    for i, entry in enumerate(corpus):
        doc, metadata = entry
        regex_doc = surround_with_undsc('___'.join(doc))

        if unique:
            # Sum the weights of unique matched words
            matched_weights = 0
            found_words = set()
            for match in pattern.finditer(regex_doc):
                found_word = match.group()
                if found_word not in found_words:
                    found_words.add(found_word)
                    matched_weights += words[found_word]

            if matched_weights >= min:
                found_docs_id.append(i)
        else:
            # Sum the weights of all matched words
            matched_weights = 0
            for match in pattern.finditer(regex_doc):
                found_word = match.group()
                matched_weights += words[found_word]

            if matched_weights >= min:
                found_docs_id.append(i)

    return found_docs_id


def corpus_from_found(
    found_docs: list[int],
    source_corpus: Corpus,
    goal_corpus='FrequencyCorpus',
):

    docs, metadata = zip(*[source_corpus[i] for i in found_docs])

    if (
        goal_corpus == 'FrequencyCorpus'
        or goal_corpus == FrequencyCorpus
    ):
        corpus = FrequencyCorpus(
            documents=docs,
            metadata=metadata,
            filter=source_corpus.filter,
            language=source_corpus.language
        )

    elif goal_corpus == 'Corpus' or goal_corpus == Corpus:
        corpus = Corpus(
            documents=docs,
            metadata=metadata,
            filter=source_corpus.filter,
            language=source_corpus.language
        )

    else:
        raise ValueError(
            "Invalid goal_corpus object."
            "Must be 'FrequencyCorpus' or 'Corpus'."
        )

    return corpus


def corpus_from_notfound(
    found_docs: list[int],
    source_corpus: Corpus,
    goal_corpus='FrequencyCorpus',
):

    docs, metadata = zip(
        *[source_corpus[i] for i in range(len(source_corpus))
          if i not in found_docs]
    )

    if (
        goal_corpus == 'FrequencyCorpus'
        or goal_corpus == FrequencyCorpus
    ):
        corpus = FrequencyCorpus(
            documents=docs,
            metadata=metadata,
            filter=source_corpus.filter,
            language=source_corpus.language
        )

    elif goal_corpus == 'Corpus' or goal_corpus == Corpus:
        corpus = Corpus(
            documents=docs,
            metadata=metadata,
            filter=source_corpus.filter,
            language=source_corpus.language
        )

    else:
        raise ValueError(
            "Invalid goal_corpus object."
            "Must be 'FrequencyCorpus' or 'Corpus'."
        )

    return corpus

import re
from ..corpus import Corpus, FrequencyCorpus
from sklearn.metrics import classification_report
from pathlib import Path
from typing import Iterable


# def _surround_with_undscs(
#     text: str,
# ) -> str:
#     return f"____{text}____"


def _surround_with_undsc(
    text: str,
) -> str:
    return f"__{text}__"


def _wordlist_edit(
    wordlist: Iterable[str | tuple]
):
    new_wordlist = []
    for ngram in wordlist:
        if isinstance(ngram, str):
            new_wordlist.append(
                _surround_with_undsc(ngram)
            )
        elif isinstance(ngram, tuple) and len(ngram) == 1:
            new_wordlist.append(
                _surround_with_undsc(ngram[0])
            )
        elif isinstance(ngram, tuple) and len(ngram) > 1:
            new_wordlist.append(
                _surround_with_undsc('____'.join(ngram))
            )
        else:
            raise ValueError(
                "Invalid wordlist format."
                "Entries must be strings or tuples."
            )
    return new_wordlist


def _worddict_edit(
    worddict: dict
):
    new_worddict = {}
    for ngram, weight in worddict.items():
        if isinstance(ngram, str):
            new_worddict[_surround_with_undsc(ngram)] = weight
        elif isinstance(ngram, tuple) and len(ngram) == 1:
            new_worddict[_surround_with_undsc(ngram[0])] = weight
        elif isinstance(ngram, tuple) and len(ngram) > 1:
            new_worddict[_surround_with_undsc('____'.join(ngram))] = weight
        else:
            raise ValueError(
                "Invalid wordlist format."
                "Entries must be strings or tuples."
            )
    return new_worddict


def _clean_matches(found_docs: dict):
    """Function to clean the matches of the wordlist
    in the corpus. Removes duplicates and empty matches."""
    cleaned_matches = {}
    for doc_id, matches in found_docs.items():
        cleaned_matches[doc_id] = tuple(
            re.sub(r'__+', ' ', match).strip()
            for match in matches if match
        )
    return cleaned_matches


def match_wordlist(
    corpus: Corpus,
    wordlist: list | set | dict,
    min: int = 1,
    unique: bool = False,
    escape=True,
    flags=0
):
    """Function to treat a list of words as a single token
    in the entire corpus."""
    words = _wordlist_edit(wordlist)
    if escape:
        escaped_words = [re.escape(word) for word in words]
    else:
        escaped_words = words
    pattern = re.compile(f"({'|'.join(escaped_words)})", flags=flags)

    found_docs_id = {}
    for i, entry in enumerate(corpus):
        doc, _ = entry
        regex_doc = _surround_with_undsc('____'.join(doc))
        if unique:
            found_words = set()
            for match in pattern.finditer(regex_doc):
                found_words.add(match.group())
            if len(found_words) >= min:
                found_docs_id[i] = tuple(found_words)
        else:
            found_words = pattern.findall(regex_doc)
            if len(found_words) >= min:
                found_docs_id[i] = tuple(set(found_words))

    return _clean_matches(found_docs_id)


def match_wordlist_pmw(
    corpus: Corpus,
    wordlist: list | set | dict,
    min_pmw: int = 1000,
    unique: bool = False,
    escape: bool = True,
    flags=0
):
    """Function to find documents in a corpus where the frequency of words
    from a wordlist exceeds a minimum parts per million threshold."""
    words = _wordlist_edit(wordlist)
    if escape:
        escaped_words = [re.escape(word) for word in words]
    else:
        escaped_words = words
    pattern = re.compile(f"({'|'.join(escaped_words)})", flags=flags)

    found_docs_id = {}
    for i, entry in enumerate(corpus):
        doc, _ = entry
        regex_doc = _surround_with_undsc('____'.join(doc))

        if unique:
            found_words = set()
            for match in pattern.finditer(regex_doc):
                found_words.add(match.group())
            word_count = len(found_words)
        else:
            found_words = pattern.findall(regex_doc)
            word_count = len(found_words)

        pmw_score = (word_count * 1000000) / len(doc)

        if pmw_score >= min_pmw:
            found_docs_id[i] = tuple(set(found_words))

    return _clean_matches(found_docs_id)


def match_regex(
    corpus: Corpus,
    regex: str | re.Pattern,
    text_key: str = 'text_deduped',
    min: int = 1,
    unique: bool = False,
):

    found_docs_id = {}
    for i, entry in enumerate(corpus):

        _, metadata = entry

        try:
            text = metadata.get(text_key, '')
        except KeyError:
            raise KeyError(
                f"Key '{text_key}' not found in metadata."
            )

        if not isinstance(text, str):
            raise ValueError(
                f"Text must be a string, but got {type(text)}."
            )

        hits = re.findall(regex, text)
        if unique:
            hits = set(hits)
        if len(hits) >= min:
            found_docs_id[i] = tuple(hits)

    return _clean_matches(found_docs_id)


def match_weighted_wordlist(
    corpus: Corpus,
    wordlist: dict,
    min: int = 1,
    unique: bool = False,
):
    """Function to treat a weighted list of words as a single token
    in the entire corpus. Returns documents where the sum of weights
    of matched words is at least equal to min."""

    words = _worddict_edit(wordlist)
    escaped_words = [re.escape(word) for word in words]
    pattern = re.compile(f"({'|'.join(escaped_words)})")

    found_docs_id = {}
    for i, entry in enumerate(corpus):
        doc, metadata = entry
        regex_doc = _surround_with_undsc('____'.join(doc))

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
                found_docs_id[i] = tuple(found_words)
        else:
            # Sum the weights of all matched words
            matched_weights = 0
            found_words = pattern.findall(regex_doc)
            for found_word in found_words:
                matched_weights += words[found_word]
            if matched_weights >= min:
                found_docs_id[i] = tuple(set(found_words))

    return _clean_matches(found_docs_id)


def corpus_from_found(
    found_docs: list[int] | dict,
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


def corpus_from_nonannotated(
    found_docs: list[int] | dict,
    source_corpus: Corpus,
    annotator: str,
    goal_corpus='FrequencyCorpus',
):
    found_docs_nonannotated = []
    for i, entry in enumerate(source_corpus):
        _, metadata = entry
        if not metadata.get(annotator, False) and i in found_docs:
            found_docs_nonannotated.append(i)

    corpus = corpus_from_found(
        found_docs_nonannotated,
        source_corpus,
        goal_corpus
    )

    return corpus


def corpus_from_notfound(
    found_docs: list[int] | dict,
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


def corpus_from_keys(
    keys: Iterable,
    source_corpus: Corpus,
    key_name: str | int = 'file',
    goal_corpus='FrequencyCorpus',
):
    data = list(source_corpus)

    found_docs = []
    keys = set(keys)
    for i, entry in enumerate(data):
        _, metadata = entry
        if metadata.get(key_name, False) in keys:
            found_docs.append(i)

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


def eval_retrieval(
    corpus: Corpus,
    found_docs: list[int] | dict,
    annotator: str,
    mode: str = 'pooling'
):

    gold_classification_main = []
    gold_classification_side = []
    retrieved_classification = []

    # Only check Hauptthema first
    for i, entry in enumerate(corpus):
        _, metadata = entry
        if mode == 'annotated':
            if not metadata.get(annotator, False):
                continue
        gold_classification_main.append(
            int(metadata.get(annotator) == '1_hauptthema')
        )
        gold_classification_side.append(
            int(metadata.get(annotator) in {
                '1_hauptthema',
                '2_nebenthema',
                '2_erwähnung'
            })
        )
        if i in found_docs:
            retrieved_classification.append(1)
        else:
            retrieved_classification.append(0)

    # Print classification report
    print('Classification report for main topic:')
    print(
        classification_report(
            gold_classification_main,
            retrieved_classification,
            target_names=['Not Relevant', 'Relevant'],
            digits=4
        )
    )
    print()
    print('Classification report for main and side topic:')
    print(
        classification_report(
            gold_classification_side,
            retrieved_classification,
            target_names=['Not Relevant', 'Relevant'],
            digits=4
        )
    )

    return (
        classification_report(
            gold_classification_main,
            retrieved_classification,
            target_names=['Not Relevant', 'Relevant'],
            digits=4,
            output_dict=True
        ),
        classification_report(
            gold_classification_side,
            retrieved_classification,
            target_names=['Not Relevant', 'Relevant'],
            digits=4,
            output_dict=True
        )
    )


def keep_keys(
    dict_: dict,
    keys: list[str]
):
    """Function to keep only the keys in a dictionary."""
    return {
        key: dict_[key]
        for key in keys
        if key in dict_
    }


def prepare_annotations(
    corpus: Corpus,
    found_docs: dict,
    annotator: str,
    goalpath: str,
    confimation: bool = True
):

    files_to_write = {}
    for i, entry in enumerate(corpus):
        _, metadata = entry
        if metadata.get(annotator, False):
            continue
        if i not in found_docs:
            continue

        relevant_data = keep_keys(
            metadata,
            ['h1', 'url', 'text', 'og:description', 'source', 'file']
        )

        goal_text = (
            f"URL: {relevant_data['url']}\n"
            f"Title: {relevant_data['h1']}\n"
            f"Description: {relevant_data['og:description']}\n"
            f"Search terms: {', '.join(list(found_docs[i]))}\n\n"
            f'++++++++++++++++++++++++++++++\n\n'
            f"{relevant_data['text']}\n"
        )

        goaldir = Path(goalpath) / 'corpus_full'
        if not goaldir.exists():
            goaldir = Path(goalpath)
        filename = f"{Path(relevant_data['file']).stem}.txt"
        filepath = Path(goaldir) / filename
        if filepath.exists():
            continue

        files_to_write[filepath] = goal_text

    if confimation:
        if input(
            f'This process will create {len(files_to_write)} files '
            f'in {goaldir}.\n'
            'Do you want to continue? (y/n): '
        ).lower() != 'y':
            print('Process cancelled.')
            return

    for filename, goal_text in files_to_write.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(goal_text)

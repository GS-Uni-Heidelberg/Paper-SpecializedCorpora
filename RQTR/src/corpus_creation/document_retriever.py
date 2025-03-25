from ..corpus import Corpus


def match_wordlist(
    corpus: Corpus,
    wordlist: list | set,
    min: int = 1,
    unique: bool = False,
):
    """Function to treat a list of words as a single token
    in the entire corpus."""
    wordlist = set(wordlist)

    found_docs = []
    for doc, metadata in corpus:
        count = 0
        seen = set()
        for word in doc:
            if unique and word in seen:
                continue
            seen.add(word)
            if word in wordlist:
                count += 1
        if count >= min:
            found_docs.append((doc, metadata))

    return found_docs


def match_weighted_wordlist(
    corpus: Corpus,
    wordlist: dict,
    min: int = 1,
    unique: bool = False,
):
    found_docs = []

    for doc, metadata in corpus:
        count = 0
        seen = set()
        for word in doc:
            if unique and word in seen:
                continue
            seen.add(word)
            if word in wordlist:
                count += wordlist[word]
        if count >= min:
            found_docs.append((doc, metadata))

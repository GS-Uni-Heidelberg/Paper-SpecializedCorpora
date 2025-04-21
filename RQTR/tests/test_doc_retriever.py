from ..src.corpus import Corpus
from ..src.corpus_creation import document_retriever as dr


fake_data_1 = [
    ['find', 'this'],
    ['not', 'this'],
    ['and', 'this', 'and', 'this']
]

CORPUS1 = Corpus(fake_data_1)


def test_wordlist_match():
    wordlist = ['find', ('this',), ('and', 'this')]
    hits = dr.match_wordlist(CORPUS1, wordlist, min=2)
    assert list(hits.keys()) == [0, 2]

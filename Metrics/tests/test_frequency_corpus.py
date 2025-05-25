from ..src.corpus import FrequencyCorpus


MOCK_DOCS = [
    ['This', 'is', 'a', 'test', 'document', '.'],
    ['this', 'is', 'another', 'test', 'document', '234'],
    ['This', 'is', 'yet', 'another', 'test', 'document', '.'],
    ['Another', 'test', 'document', 'This', 'is', '!'],
]


def test_basic_freqs():
    mock_corpus = FrequencyCorpus(MOCK_DOCS, filter=None)
    assert mock_corpus.get_ngrams(1) == {
        ('This',): 3,
        ('is',): 4,
        ('a',): 1,
        ('test',): 4,
        ('document',): 4,
        ('.',): 2,
        ('this',): 1,
        ('another',): 2,
        ('234',): 1,
        ('yet',): 1,
        ('Another',): 1,
        ('!',): 1,
    }
    assert mock_corpus.get_ngrams(2)[('This', 'is')] == 3
    assert mock_corpus.get_ngrams(2)[('is', 'a')] == 1

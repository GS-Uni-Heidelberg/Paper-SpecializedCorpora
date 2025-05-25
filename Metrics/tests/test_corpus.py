from ..src.corpus import Corpus


MOCK_DOCS = [
    ['This', 'is', 'a', 'test', 'document', '.'],
    ['this', 'is', 'another', 'test', 'document', '234'],
    ['This', 'is', 'yet', 'another', 'test', 'document', '.'],
    ['Another', 'test', 'document', 'This', 'is', '!'],
]


def test_filter():
    mock_corpus1 = Corpus(MOCK_DOCS, filter=None)
    assert mock_corpus1.documents == MOCK_DOCS

    mock_corpus2 = Corpus(MOCK_DOCS, filter=lambda x, y: x[0].islower())
    print(mock_corpus2.documents)
    assert mock_corpus2.documents[0] == [
        'is', 'a', 'test', 'document'
    ]


def test_treat_as_one():
    mock_corpus1 = Corpus(MOCK_DOCS, filter=None)
    mock_corpus1.treat_as_one(['test', 'document'], 'test_document')
    assert mock_corpus1.documents == [
        ['This', 'is', 'a', 'test_document', '.'],
        ['this', 'is', 'another', 'test_document', '234'],
        ['This', 'is', 'yet', 'another', 'test_document', '.'],
        ['Another', 'test_document', 'This', 'is', '!'],
    ]

    mock_corpus2 = Corpus(MOCK_DOCS, filter=None)
    mock_corpus2.treat_as_one(['This', 'is'])
    assert mock_corpus2.documents == [
        ['This is', 'a', 'test', 'document', '.'],
        ['this', 'is', 'another', 'test', 'document', '234'],
        ['This is', 'yet', 'another', 'test', 'document', '.'],
        ['Another', 'test', 'document', 'This is', '!'],
    ]

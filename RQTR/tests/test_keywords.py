from ..src.corpus import FrequencyCorpus
from ..src.metrics import keyness


MOCK_DOCS = [
    ['This', 'is', 'a', 'test', 'document', '.'],
    ['this', 'is', 'another', 'test', 'document', '234'],
    ['This', 'is', 'yet', 'another', 'test', 'document', '.'],
    ['Another', 'test', 'document', 'This', 'is', '!'],
]

MOCH_REFERENCE = [
    ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'],
    ['Lorem', 'ipsum', 'dolor', 'sit', 'amet', ',', 'consectetur', 'adipiscing', 'elit', '.'],
    ['Hello', 'world', '!', 'This', 'is', 'a', 'test', 'document', '.'],
]

STUDY_CORPUS = FrequencyCorpus(MOCK_DOCS, filter=None)
REFERENCE_CORPUS = FrequencyCorpus(MOCH_REFERENCE, filter=None)


def test_contingency():
    contingency = keyness.corpus_to_contingency(
        ('This',), STUDY_CORPUS, REFERENCE_CORPUS,
        smoothing=0
    )

    assert contingency[0, 0] == 3
    assert contingency[0, 1] == 1
    assert contingency[1, 0] == 22

    contingency2 = keyness.corpus_to_contingency(
        ('test', 'document'), STUDY_CORPUS, REFERENCE_CORPUS,
        smoothing=1, filter=lambda x, y: x[0].islower()
    )

    assert contingency2[0, 0] == 5
    assert contingency2[1, 0] == 15

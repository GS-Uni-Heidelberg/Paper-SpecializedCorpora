from ..src.corpus import FrequencyCorpus
from ..src.metrics import keyness
import numpy as np


MOCK_DOCS = [
    ['This', 'is', 'a', 'test', 'document', '.'],
    ['this', 'is', 'another', 'test', 'document', '234'],
    ['This', 'is', 'yet', 'another', 'test', 'document', '.'],
    ['Another', 'test', 'document', 'This', 'is', '!'],
]

MOCK_REFERENCE = [
    [
        'The', 'quick', 'brown', 'fox',
        'jumps', 'over', 'the', 'lazy', 'dog', '.'
    ],
    [
        'Lorem', 'ipsum', 'dolor', 'sit', 'amet', ',',
        'consectetur', 'adipiscing', 'elit', '.'
    ],
    ['Hello', 'world', '!', 'This', 'is', 'a', 'test', 'document', '.'],
]

STUDY_CORPUS = FrequencyCorpus(MOCK_DOCS, filter=None)
REFERENCE_CORPUS = FrequencyCorpus(MOCK_REFERENCE, filter=None)


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


def test_statistics():
    contingency = np.array([[2, 1], [8, 9]])

    assert keyness.odds_ratio(contingency) == 2.25
    assert keyness.percent_difference(contingency) == 100

    contingency2 = np.array([[10, 5], [5, 10]])
    ll_scipy = keyness.log_likelihood_scipy(contingency2)
    assert ll_scipy > 2 and ll_scipy < 2.2

    ll_rayson = keyness.log_likelihood_rayson(contingency2)
    assert ll_rayson > 1.6 and ll_rayson < 1.8  # According to the online tool

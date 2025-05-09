import pytest
from ..src.metrics import cooccurrence
from ..src.metrics import cooccurrence_parallel
from ..src.corpus import Corpus


MOCK_DOCS = [
    ['this', 'is', 'a', 'test', 'document', '.'],
    [
        'this', 'is', 'another', 'test', 'document', '.',
        'this', 'is', 'second', 'sentence', '.'
    ],
    ['yet', 'another', 'test', 'document', '.'],
]


MODULES = [
    cooccurrence,
    cooccurrence_parallel
]


@pytest.mark.parametrize("module", MODULES)
def test_cooccurrence_basic(module):
    corpus = Corpus(MOCK_DOCS, filter=None)

    cooccurrences = module.Cooccurrences(
        window_size=2,
        unit_separator=None
    )
    cooccurrences.count_cooccurrences(
        corpus
    )

    table = cooccurrences.cooccurrence_table
    assert table['this']['is'] == 3
    assert table['is']['this'] == 3
    assert table['test']['.'] == 3
    assert table['yet']['test'] == 1
    assert table['this']['document'] == 1


@pytest.mark.parametrize("module", MODULES)
def test_cooccurrence_smooth(module):
    corpus = Corpus(MOCK_DOCS, filter=None)

    cooccurrences = module.Cooccurrences(
        window_size=2,
        unit_separator=None,
        smoothing=0.5
    )
    cooccurrences.count_cooccurrences(
        corpus
    )

    table = cooccurrences.cooccurrence_table
    assert table['this']['is'] == 3.5
    assert table['is']['this'] == 3.5
    assert table['yet']['.'] == 0.5


@pytest.mark.parametrize("module", MODULES)
def test_cooccurrence_unit_separator(module):
    corpus = Corpus(MOCK_DOCS, filter=None)

    cooccurrences = module.Cooccurrences(
        window_size=2,
        unit_separator='.'
    )
    cooccurrences.count_cooccurrences(
        corpus
    )

    table = cooccurrences.cooccurrence_table
    assert table['this']['is'] == 3
    assert table['is']['this'] == 3
    assert table['test'].get('.') is None
    assert table['document'].get('this') is None


@pytest.mark.parametrize("module", MODULES)
def test_cooccurrence_document(module):
    corpus = Corpus(MOCK_DOCS, filter=None)

    cooccurrences = module.Cooccurrences(
        window_size=None,
        unit_separator=None
    )
    cooccurrences.count_cooccurrences(
        corpus
    )

    table = cooccurrences.cooccurrence_table
    assert table['this']['.'] == 3
    assert table['yet']['.'] == 1


@pytest.mark.parametrize("module", MODULES)
def test_cooccurrence_duplicate_counting(module):
    corpus = Corpus(MOCK_DOCS, filter=None)

    cooccurrences = module.Cooccurrences(
        window_size=None,
        unit_separator=None,
        duplicate_counting=False
    )
    cooccurrences.count_cooccurrences(
        corpus
    )

    table = cooccurrences.cooccurrence_table
    assert table['this']['is'] == 2
    assert table['another']['this'] == 1

import math
import pandas as pd
from typing import Callable


class BaseCooccurrences():
    """Class to count cooccurrences of words in a corpus."""

    def __init__(
        self,
        window_size: int | None = 5,
        unit_separator: str | None = None,
        smoothing: float | None = None,
        duplicate_counting: bool = True
    ):
        """Initialize the cooccurrence table with the params
        used to count the cooccurrences.

        If smoothing is provided, the cooccurrence table is smoothed
        by adding the smoothing parameter to each cell.

        Parameters:
            window_size (int | None): The size of the window to use to count
                cooccurrences. If None, the whole document or unit is used.
                Defaults to 5.
            unit_separator (str | None): If a unit_separator is provided, the
                document is split into units using the separator. Can be used
                e.g. to calculate cooccurrences paragraph-wide.
                Defaults to None.
            smoothing (float): The smoothing parameter to use when calculating
                the cooccurrence table. Defaults to 0.0.
        """
        self.window_size = window_size
        self.unit_separator = unit_separator
        self.smoothing = smoothing
        self._total_collocations = None
        self.cooccurrence_table = None
        self.duplicate_counting = duplicate_counting
        self.vocab = set()


def calculate_pmi(
    word1: str,
    word2: str,
    collocations: BaseCooccurrences,
    smoothing: float = 0.0,
    exp: float = 1,
    normalize: bool = False
) -> float:
    """
    Calculate the pointwise mutual information (PMI) score for a pair of words
    using the CooccurrenceTable.

    Args:
        word1: First word.
        word2: Second word.
        collocations: CooccurrenceTable object
            containing the cooccurrence matrix.
        smoothing: Smoothing parameter to avoid zero counts, log(0)
            and generally bias towards smaller values.
        exp: Exponent to raise the PMI score to.
            Default is 1 (no exponentiation, classic PMI).
            Increasing the exponent will result in more general
            collocations achieving higher scores.

    Returns:
        float: Pointwise Mutual Information (PMI) score.
    """

    # Check if both words are in the vocabulary
    if word1 not in collocations.vocab or word2 not in collocations.vocab:
        return float('-inf')

    cooccurrence_table = collocations.cooccurrence_table

    # Get counts with smoothing
    f_w1_r_w2 = cooccurrence_table[word1].get(word2, 0) + smoothing
    f_w1_r = (
        cooccurrence_table[word1]['__total__']
        + smoothing * len(collocations.vocab)
    )
    f_r_w2 = (
        cooccurrence_table[word2]['__total__']
        + smoothing * len(collocations.vocab)
    )
    f_total = (
        collocations.total_collocations
        + smoothing * len(collocations.vocab) ** 2
    )

    # Avoid division by zero / log(0)
    if f_w1_r == 0 or f_r_w2 == 0 or f_w1_r_w2 == 0 or f_total == 0:
        return float('-inf')

    # Calculate PMI score with correct exponentiation
    pmi_score = math.log2(
        (f_w1_r_w2**exp * f_total)
        /
        (f_w1_r * f_r_w2)
    )

    # Normalize if requested
    if normalize:
        p_w1_w2 = f_w1_r_w2 / f_total
        denominator = -math.log2(p_w1_w2)  # Correct normalization factor
        if denominator != 0:
            pmi_score = pmi_score / denominator

    return pmi_score


def calculate_logdice(
    word1: str,
    word2: str,
    collocations: BaseCooccurrences,
    smoothing: float = 0.0
) -> float:
    """
    Calculate the logDice score for a pair of words using CooccurrenceTable.

    Args:
        word1: First word
        word2: Second word
        collocations: CooccurrenceTable object
            containing the cooccurrence matrix

    Returns:
        float: logDice score
    """
    # Access the cooccurrence table (Pandas DataFrame)
    cooccurrence_table = collocations.cooccurrence_table

    # Check if both words are in the vocabulary
    if word1 not in collocations.vocab or word2 not in collocations.vocab:
        return float('-inf')  # Return -inf if either word is not in the table

    # ||w1, R, w2|| - frequency of the specific collocation
    f_w1_r_w2 = (
        cooccurrence_table[word1].get(word2, 0)
        + smoothing
    )

    # ||w1, R, *|| - frequency of first word with any relation (row sum)
    f_w1_r = (
        cooccurrence_table[word1]['__total__']
        + smoothing * len(collocations.vocab)
    )

    # ||*, R, w2|| - frequency of second word with any relation (column sum)
    f_r_w2 = (
        cooccurrence_table[word2]['__total__']
        + smoothing * len(collocations.vocab)
    )

    # Avoid division by zero
    if f_w1_r == 0 or f_r_w2 == 0 or f_w1_r_w2 == 0:
        return float('-inf')

    # Calculate logDice score
    dice_score = (2 * f_w1_r_w2) / (f_w1_r + f_r_w2)
    logdice_score = 14 + math.log2(dice_score)

    return logdice_score


def calculate_minsens(
    word1: str,
    word2: str,
    collocations: BaseCooccurrences,
    smoothing: float = 0.0
) -> float:
    """
    Calculate the minimum sensitivity score for a pair of words using
    CooccurrenceTable.

    Args:
        word1: First word
        word2: Second word
        collocations: CooccurrenceTable object
            containing the cooccurrence matrix

    Returns:
        float: Minimum sensitivity score
    """

    # Access the cooccurrence table (Pandas DataFrame)
    cooccurrence_table = collocations.cooccurrence_table

    def calculate_sensitivity(focus_word, other_word):
        """Calculate the sensitivity of a word pair."""

        # Avoid division by zero
        if (
            cooccurrence_table[focus_word]['__total__'] == 0
            or cooccurrence_table[focus_word].get(other_word, 0) == 0
        ):
            return float('-inf')

        return (
            (
                cooccurrence_table[focus_word].get(other_word, 0)
                + smoothing
            )
            /
            (
                cooccurrence_table[focus_word]['__total__']
                + smoothing * len(collocations.vocab)
            )
        )

    sensitivity1 = calculate_sensitivity(word1, word2)
    sensitivity2 = calculate_sensitivity(word2, word1)
    min_sensitivity = min(sensitivity1, sensitivity2)

    return min_sensitivity


def all_collocations(
    cooccurrences: BaseCooccurrences,
    word: str,
    method: Callable | str,
    min_count: int = 0,
    **kwargs
):

    if isinstance(method, str):
        if method in {'pmi', 'calculate_pmi'}:
            method = calculate_pmi
        elif method in {'logdice', 'calculate_logdice'}:
            method = calculate_logdice
        elif method in {'minsens', 'calculate_minsens'}:
            method = calculate_minsens
        else:
            raise ValueError('Invalid Method String: Not supported.')

    all_results = [
        (other_term, method(word, other_term, cooccurrences, **kwargs))
        for other_term in cooccurrences.vocab
        if cooccurrences.cooccurrence_table[word].get(
            other_term, 0
        ) >= min_count
    ]

    df = pd.DataFrame(all_results, columns=['Term', 'Stat'])
    df.sort_values('Stat', inplace=True, ascending=False)

    return df

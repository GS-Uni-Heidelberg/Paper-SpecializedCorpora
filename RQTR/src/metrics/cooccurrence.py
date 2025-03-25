from ..corpus import Corpus
import pandas as pd
from tqdm import tqdm
import math
from collections import defaultdict
from dataclasses import dataclass


class Cooccurrences():
    def __init__(
        self,
        window_size: int | None = 5,
        unit_separator: str | None = None,
        smoothing: float | None = None,
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
        self.vocab = set()

    def count_cooccurrences(self, corpus: Corpus):
        """Count the cooccurrences of words in the corpus.
        """

        cooccurrence_table = defaultdict(lambda: defaultdict(0))
        for document in tqdm(corpus.documents):
            if self.unit_separator:
                units = [
                    unit.split()
                    for unit in document.split(self.unit_separator)
                ]
            else:
                units = [document]

            for unit in units:
                for i, word in enumerate(unit):
                    seen_words = set()
                    if word not in cooccurrence_table:
                        cooccurrence_table[word] = {}

                    if self.window_size:
                        for j in range(
                            i - self.window_size, i + self.window_size + 1
                        ):
                            if j < 0 or j >= len(unit) or j == i:
                                continue
                            if unit[j] in seen_words:  # Skip duplicate words
                                continue
                            seen_words.add(unit[j])

                            cooccurrence_table[word][unit[j]] = cooccurrence_table[
                                word
                            ].get(unit[j], 0) + 1
                    else:
                        for other_word in unit:
                            if other_word in seen_words:
                                continue
                            seen_words.add(other_word)
                            cooccurrence_table[word][other_word] = cooccurrence_table[
                                word
                            ].get(other_word, 0) + 1

        self.cooccurrence_table = cooccurrence_table
        self.vocab = set(cooccurrence_table.keys())
        self.apply_smoothing()
        self.__margin_sums()
        self.calc_total_collocations()

    def __margin_sums(self):
        """Calculate the row and column sums of the cooccurrence table and
        add it under the '__total__' key.
        """

        for cooccurrences in self.cooccurrence_table.values():
            cooccurrences['__total__'] = sum(cooccurrences.values())

    def apply_smoothing(self):
        """Apply the smoothing parameter to the cooccurrence table.
        """
        if not self.smoothing:
            return

        for cooccurrences in self.cooccurrence_table.values():
            for word in self.vocab:
                cooccurrences[word] = (
                    cooccurrences.get(word, 0) + self.smoothing
                )

        return

    def undo_smoothing(self):
        """Undo the smoothing parameter from the cooccurrence table.
        """

        if not self.smoothing:
            return

        for cooccurrences in self.cooccurrence_table.values():
            for word in self.vocab:
                cooccurrences[word] = (
                    cooccurrences.get(word, 0) - self.smoothing
                )

    def pop_stop(self, stop_words: list[str]):
        """Remove stop words from the cooccurrence table.
        """

        stop_words = set(stop_words)

        for stop_word in stop_words:
            if stop_word in self.cooccurrence_table:
                self.cooccurrence_table.pop(stop_word)
        for cooccurrences in self.cooccurrence_table.values():
            for stop_word in stop_words:
                if stop_word in cooccurrences:
                    cooccurrences.pop(stop_word)
        self.calc_total_collocations()
        self.vocab = set(self.cooccurrence_table.columns)

    @property
    def total_collocations(self):
        """Return the total number of collocations in the table.
        """
        if not self._total_collocations:
            self.calc_total_collocations()
        return self._total_collocations

    def calc_total_collocations(self):
        """Calculate the total number of collocations in the table.
        """
        total = 0
        for cooccurrences in self.cooccurrence_table.values():
            total += cooccurrences['__total__']

        self._total_collocations = total


def calculate_pmi(
    word1: str,
    word2: str,
    collocations: Cooccurrences,
    smoothing: float = 0.0
) -> float:
    """
    Calculate the pointwise mutual information (PMI) score for a pair of words
    using the CooccurrenceTable.

    Args:
        word1: First word.
        word2: Second word.
        collocations: CooccurrenceTable object containing the cooccurrence matrix.

    Returns:
        float: Pointwise Mutual Information (PMI) score.
    """
    # Access the cooccurrence table (Pandas DataFrame)
    cooccurrence_table = collocations.cooccurrence_table

    # Check if both words are in the vocabulary
    if word1 not in collocations.vocab or word2 not in collocations.vocab:
        return float('-inf')  # Return -inf if either word is not in the table

    # ||w1, R, w2|| - frequency of the specific collocation
    f_w1_r_w2 = cooccurrence_table[word1].get(word2, 0) + smoothing

    # ||w1, R, *|| - frequency of first word with any relation (row sum)
    f_w1_r = cooccurrence_table[word1]['__total__'] + smoothing * len(collocations.vocab)

    # ||*, R, w2|| - frequency of second word with any relation (column sum)
    f_r_w2 = cooccurrence_table[word2]['__total__'] + smoothing * len(collocations.vocab)

    # ||*, *, *|| - total frequency of all collocations
    f_total = collocations.total_collocations + smoothing * len(collocations.vocab) ** 2

    # Avoid division by zero / log(0)
    if f_w1_r == 0 or f_r_w2 == 0 or f_w1_r_w2 == 0 or f_total == 0:
        return float('-inf')

    # Calculate PMI score
    pmi_score = math.log2((f_w1_r_w2 * f_total) / (f_w1_r * f_r_w2))

    return pmi_score


def calculate_logdice(
    word1: str,
    word2: str,
    collocations: Cooccurrences,
    smoothing: float = 0.0
) -> float:
    """
    Calculate the logDice score for a pair of words using the CooccurrenceTable.

    Args:
        word1: First word
        word2: Second word
        collocations: CooccurrenceTable object containing the cooccurrence matrix

    Returns:
        float: logDice score
    """
    # Access the cooccurrence table (Pandas DataFrame)
    cooccurrence_table = collocations.cooccurrence_table

    # Check if both words are in the vocabulary
    if word1 not in collocations.vocab or word2 not in collocations.vocab:
        return float('-inf')  # Return -inf if either word is not in the table

    # ||w1, R, w2|| - frequency of the specific collocation
    f_w1_r_w2 = cooccurrence_table[word1].get(word2, 0) + smoothing

    # ||w1, R, *|| - frequency of first word with any relation (row sum)
    f_w1_r = cooccurrence_table[word1]['__total__'] + smoothing * len(collocations.vocab)

    # ||*, R, w2|| - frequency of second word with any relation (column sum)
    f_r_w2 = cooccurrence_table[word2]['__total__'] + smoothing * len(collocations.vocab)

    # Avoid division by zero
    if f_w1_r == 0 or f_r_w2 == 0 or f_w1_r_w2 == 0:
        return float('-inf')

    # Calculate logDice score
    dice_score = (2 * f_w1_r_w2) / (f_w1_r + f_r_w2)
    logdice_score = 14 + math.log2(dice_score)

    return logdice_score

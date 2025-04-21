from ..corpus import Corpus
from tqdm import tqdm
import math
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict, Counter
from functools import partial


# +++ FUNCTIONS FOR PARALLEL PROCESSING +++ #
# Outside the class to avoid pickling issues etc.

def _split_document_into_units(
    document: list[str],
    unit_separator: str | None = None
):
    """Split a document into units based on the unit_separator."""
    if not unit_separator:
        return [document]
    split_doc = []
    unit = []
    for word in document:
        if word == unit_separator:
            if unit:
                split_doc.append(unit)
                unit = []
        else:
            unit.append(word)
    if len(unit) > 0:
        split_doc.append(unit)
    return split_doc


def _process_document(
    document: list[str],
    window_size: int | None,
    unit_separator: str | None
):
    """Process a single document in parallel"""
    units = _split_document_into_units(document, unit_separator)

    local_cooccurrences = defaultdict(Counter)

    for unit in units:
        for i, word in enumerate(unit):
            seen_words = set()

            if window_size:
                # Windowed cooccurrences
                start_idx = max(0, i - window_size)
                end_idx = min(len(unit), i + window_size + 1)

                for j in range(start_idx, end_idx):
                    if j == i or unit[j] in seen_words:
                        continue
                    seen_words.add(unit[j])
                    local_cooccurrences[word][unit[j]] += 1
            else:
                # Unit-wide cooccurrences
                for other_word in unit:
                    if other_word == word or other_word in seen_words:
                        continue
                    seen_words.add(other_word)
                    local_cooccurrences[word][other_word] += 1

    return local_cooccurrences


def _merge_cooccurrences(results):
    """Merge cooccurrence results from parallel processing"""
    merged = defaultdict(lambda: defaultdict(int))
    for local_cooccurrences in tqdm(
        results,
        desc="Merging cooccurrences"
    ):
        for word, counter in local_cooccurrences.items():
            for other_word, count in counter.items():
                merged[word][other_word] += count
    return merged


class Cooccurrences():
    """Class to count cooccurrences of words in a corpus."""

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

    def count_cooccurrences(
        self,
        corpus: Corpus,
        max_workers=None
    ):
        """Count the cooccurrences of words in the corpus
        using parallel processing."""
        process_doc = partial(
            _process_document,
            window_size=self.window_size,
            unit_separator=self.unit_separator
        )

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Using tqdm to show progress
            results = list(tqdm(
                executor.map(process_doc, corpus.documents),
                total=len(corpus.documents),
                desc="Processing documents"
            ))

        # Merge results from all processes
        self.cooccurrence_table = _merge_cooccurrences(results)

        self.vocab = set(self.cooccurrence_table.keys())
        self.apply_smoothing()
        self._calculate_margin_sums()
        self.calc_total_collocations()

    def _process_units(
        self,
        units: list[str]
    ):
        """Process all units to count cooccurrences."""
        for unit in units:
            for i, word in enumerate(unit):
                self._process_word_cooccurrences(word, unit, i)

    def _process_word_cooccurrences(self, word, unit, word_index):
        """Process a single word's cooccurrences within a unit."""
        seen_words = set()

        if self.window_size:
            self._count_windowed_cooccurrences(
                word, unit, word_index, seen_words
            )
        else:
            self._count_unit_wide_cooccurrences(
                word, unit, seen_words
            )

    def _count_windowed_cooccurrences(
        self,
        word: str,
        unit: list[str],
        word_index: int,
        seen_words: set[str]
    ):
        """Count cooccurrences within a window around the word."""
        start_idx = max(0, word_index - self.window_size)
        end_idx = min(len(unit), word_index + self.window_size + 1)

        for j in range(start_idx, end_idx):
            if j == word_index or unit[j] in seen_words:
                continue

            seen_words.add(unit[j])
            self._add_cooccurrence(word, unit[j])

    def _count_unit_wide_cooccurrences(
        self,
        word: str,
        unit: list[str],
        seen_words: set[str]
    ):
        """Count cooccurrences across the entire unit."""
        for other_word in unit:
            if other_word == word or other_word in seen_words:
                continue

            seen_words.add(other_word)
            self._add_cooccurrence(word, other_word)

    def _add_cooccurrence(
        self,
        word: str,
        other_word: str
    ):
        """Increment the cooccurrence count for a word pair."""
        self.cooccurrence_table[word][other_word] = (
            self.cooccurrence_table[word].get(other_word, 0)
            + 1
        )

    def _calculate_margin_sums(self):
        """Calculate the row and column sums of the cooccurrence table."""
        for cooccurrences in self.cooccurrence_table.values():
            cooccurrences['__total__'] = sum(cooccurrences.values())

    def apply_smoothing(self):
        """Apply the smoothing parameter to the cooccurrence table."""
        if not self.smoothing:
            return

        for cooccurrences in self.cooccurrence_table.values():
            for word in self.vocab:
                cooccurrences[word] = (
                    cooccurrences.get(word, 0)
                    + self.smoothing
                )

    def undo_smoothing(self):
        """Undo the smoothing parameter from the cooccurrence table."""
        if not self.smoothing:
            return

        for cooccurrences in self.cooccurrence_table.values():
            for word in self.vocab:
                cooccurrences[word] = (
                    cooccurrences.get(word, 0)
                    - self.smoothing
                )

    def pop_stop(self, stop_words: list[str]):
        """Remove stop words from the cooccurrence table."""
        stop_words = set(stop_words)

        for stop_word in stop_words:
            if stop_word in self.cooccurrence_table:
                self.cooccurrence_table.pop(stop_word)

        for cooccurrences in self.cooccurrence_table.values():
            for stop_word in stop_words:
                if stop_word in cooccurrences:
                    cooccurrences.pop(stop_word)

        self.calc_total_collocations()
        self.vocab = set(self.cooccurrence_table.keys())

    @property
    def total_collocations(self):
        """Return the total number of collocations in the table."""
        if not self._total_collocations:
            self.calc_total_collocations()
        return self._total_collocations

    def calc_total_collocations(self):
        """Calculate the total number of collocations in the table."""
        total = 0
        for cooccurrences in self.cooccurrence_table.values():
            total += cooccurrences['__total__']
        self._total_collocations = total


def calculate_pmi(
    word1: str,
    word2: str,
    collocations: Cooccurrences,
    smoothing: float = 0.0,
    exp: float = 1
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

    # ||*, *, *|| - total frequency of all collocations
    f_total = (
        collocations.total_collocations
        + smoothing * len(collocations.vocab) ** 2
    )

    # Avoid division by zero / log(0)
    if f_w1_r == 0 or f_r_w2 == 0 or f_w1_r_w2 == 0 or f_total == 0:
        return float('-inf')

    # Calculate PMI score
    pmi_score = math.log2(
        ((f_w1_r_w2 * f_total)**exp)
        /
        (f_w1_r * f_r_w2)
    )

    return pmi_score


def calculate_logdice(
    word1: str,
    word2: str,
    collocations: Cooccurrences,
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
    collocations: Cooccurrences,
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

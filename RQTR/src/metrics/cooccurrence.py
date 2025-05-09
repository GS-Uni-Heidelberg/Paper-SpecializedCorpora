from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict, Counter
from functools import partial
from ._cooccurrence_shared import (
    calculate_logdice,
    calculate_minsens,
    calculate_pmi,
    all_collocations,
    BaseCooccurrences
)

__all__ = [
    'Cooccurrences',
    'calculate_logdice',
    'calculate_minsens',
    'calculate_pmi',
    'all_collocations'
]


class Cooccurrences(BaseCooccurrences):
    def __init__(
        self,
        window_size: int | None = 5,
        unit_separator: str | None = None,
        smoothing: float | None = None,
    ):
        super().__init__(
            window_size=window_size,
            unit_separator=unit_separator,
            smoothing=smoothing
        )

    def process_document(self, document, window_size, unit_separator):
        """Process a single document in parallel"""
        units = self._split_document_into_units(document)

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

    def merge_cooccurrences(self, results):
        """Merge cooccurrence results from parallel processing"""
        merged = defaultdict(lambda: defaultdict(int))
        for local_cooccurrences in results:
            for word, counter in local_cooccurrences.items():
                for other_word, count in counter.items():
                    merged[word][other_word] += count
        return merged

    # Update the count_cooccurrences method
    def count_cooccurrences(self, corpus, max_workers=None):
        """Count the cooccurrences of words in the corpus
        using parallel processing."""
        process_doc = partial(
            self.process_document,
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
        self.cooccurrence_table = self.merge_cooccurrences(results)

        self.vocab = set(self.cooccurrence_table.keys())
        self.apply_smoothing()
        self._calculate_margin_sums()
        self.calc_total_collocations()

    def _split_document_into_units(self, document):
        """Split a document into units based on the unit_separator."""
        if not self.unit_separator:
            return [document]
        return [unit.split() for unit in document.split(self.unit_separator)]

    def _process_units(self, units):
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

    def _add_cooccurrence(self, word, other_word):
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

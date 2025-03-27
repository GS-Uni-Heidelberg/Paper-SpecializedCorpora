import numpy as np
import math
from scipy import stats
from ..corpus import FrequencyCorpus


def percent_difference(cont_table):
    """
    Calculate the percent difference of a word between two corpora.

    Parameters
    ----------
    word : str
        The word to calculate the percent difference for.
    study_corpus : Corpus
        The corpus to calculate the percent difference for.
    ref_corpus : Corpus
        The reference corpus to calculate the percent difference for.

    Returns
    -------
    percent_difference : float
        The percent difference of the word between the two corpora.
    """
    # Calculate the percent difference

    normal_freq_study = (
        cont_table[0, 0] /
        (cont_table[0, 0] + cont_table[1, 0])
    )
    normal_freq_ref = (
        cont_table[0, 1]
        / (cont_table[0, 1] + cont_table[1, 1])
    )

    try:
        percent_difference = (
            100 *
            (normal_freq_study - normal_freq_ref)
            / normal_freq_ref
        )
    except ZeroDivisionError:
        percent_difference = float('inf')

    return percent_difference


def log_likelihood_scipy(contingency_table):
    """
    Calculate the log likelihood of a contingency table.

    Parameters
    ----------
    contingency_table : array_like
        A 2x2 contingency table.

    Returns
    -------
    log_likelihood : float
        The log likelihood of the contingency table.
    """
    # Calculate the log likelihood
    log_likelihood = stats.chi2_contingency(
        contingency_table, lambda_="log-likelihood"
    )[0]

    return log_likelihood


def log_likelihood_rayson(contingency_table):
    study_corpus_size = contingency_table[:, 0].sum()
    ref_corpus_size = contingency_table[:, 1].sum()
    full_corpus_size = study_corpus_size + ref_corpus_size

    frequency_sum = contingency_table[0, :].sum()

    e1 = study_corpus_size * frequency_sum / full_corpus_size
    e2 = ref_corpus_size * frequency_sum / full_corpus_size

    log_likelihood = 2 * (
        contingency_table[0, 0] * math.log(contingency_table[0, 0] / e1)
        + contingency_table[0, 1] * math.log(contingency_table[0, 1] / e2)
    )

    return log_likelihood


def bayes_factor(contingency_table, ll_function=log_likelihood_rayson):

    ll = ll_function(contingency_table)

    bic = ll - math.log(np.sum(contingency_table))

    return bic


def odds_ratio(contingency_table):
    """
    Calculate the odds ratio of a contingency table.

    Parameters
    ----------
    contingency_table : array_like
        A 2x2 contingency table.

    Returns
    -------
    odds_ratio : float
        The odds ratio of the contingency table.
    """
    # Calculate the odds ratio
    odds_ratio = (
        (contingency_table[0, 0] * contingency_table[1, 1])
        / (contingency_table[0, 1] * contingency_table[1, 0])
    )

    return odds_ratio


def corpus_to_contingency(
    ngram: tuple,
    study_corpus: FrequencyCorpus,
    ref_corpus: FrequencyCorpus,
    smoothing=0.00001,
    filter=lambda x, y: True
):
    n = len(ngram)

    study_ngrams = study_corpus.get_ngrams(n, filter)
    ref_ngrams = ref_corpus.get_ngrams(n, filter)

    study_unique = study_corpus.unique[n]
    ref_unique = ref_corpus.unique[n]

    study_size = study_corpus.size[n]
    ref_size = ref_corpus.size[n]

    contingency_table = np.zeros((2, 2))

    # First row: observed ngram frequencies
    contingency_table[0, 0] = study_ngrams.get(ngram, 0) + smoothing
    contingency_table[0, 1] = ref_ngrams.get(ngram, 0) + smoothing

    # Second row: observed non-ngram frequencies
    contingency_table[1, 0] = (
        study_size - contingency_table[0, 0]
        + smoothing * study_unique
    )
    contingency_table[1, 1] = (
        ref_size - contingency_table[0, 1]
        + smoothing * ref_unique
    )

    return contingency_table

import numpy as np
import math
from scipy import stats
from ..corpus import FrequencyCorpus
from ..token_util import begin_end_stopword
from typing import Callable
import pandas as pd
import sys


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


def keyword_list_ngram(
    study_corpus: FrequencyCorpus,
    ref_corpus: FrequencyCorpus,
    metric: str | Callable,
    ngram_len: int = 1,
    min_docs: int = 1,
    min_freq: int = 1,
    filter_stopwords: bool = False,
    **kwargs
):
    """Calculate the keyness of all ngrams of len n in the study corpus.

    Parameters:
        study_corpus (FrequencyCorpus): The study corpus.
        ref_corpus (FrequencyCorpus): The reference corpus.
        metric (str or callable): The metric to use for keyness calculation.
        ngram_len (int): The length of the n-grams to study
            (1: study words, 2: bigrams, etc.).
            Default is 1.
        min_docs (int): The minimum number of documents
            a word must appear in to be included in the results.
            Default is 1.
        min_freq (int): The minimum frequency a word must have
            to be included in the results.
            ult is 3.
        **kwargs: Additional arguments to pass to the contigency_table
            function, e.g. smoothing.

    Returns:
        pd.DataFrame: A DataFrame containing the keyness scores
            and the corresponding ngrams.
    """

    if isinstance(metric, str):
        metric_function = string_to_function(metric)
    else:
        metric_function = metric

    keynesses = {}
    for word in study_corpus.get_ngrams(ngram_len):
        if study_corpus.ngram_doccounts[ngram_len][word] < min_docs:
            continue
        if study_corpus.ngram_counts[ngram_len][word] < min_freq:
            continue
        if (
            filter_stopwords
            and begin_end_stopword(word, study_corpus.language)
        ):
            continue
        cont_table = corpus_to_contingency(
            word,
            study_corpus,
            ref_corpus,
            **kwargs
        )

        keyness_score = metric_function(cont_table)
        keynesses[word] = keyness_score

    df = pd.DataFrame(keynesses.items(), columns=['Word', 'Keyness'])
    df = df.sort_values(by='Keyness', ascending=False)

    return df


def keyword_list(
    study_corpus: FrequencyCorpus,
    ref_corpus: FrequencyCorpus,
    metric: str | Callable,
    max_ngram_len: int = 3,
    min_docs: int = 1,
    min_freq: int = 1,
    filter_stopwords: bool = False,
    **kwargs
):
    """Calculate the keyness of all ngrams of len up to and including n
    in the study corpus.

    Parameters:
        study_corpus (FrequencyCorpus): The study corpus.
        ref_corpus (FrequencyCorpus): The reference corpus.
        metric (str or callable): The metric to use for keyness calculation.
        max_ngram_len (int): Up to this n-gram length, the list
            of keyness will be calculated.
            Default is 3.
        min_docs (int): The minimum number of documents
            a word must appear in to be included in the results.
            Default is 1.
        min_freq (int): The minimum frequency a word must have
            to be included in the results.
            ult is 3.
        **kwargs: Additional arguments to pass to the contigency_table
            function, e.g. smoothing.

    Returns:
        pd.DataFrame: A DataFrame containing the keyness scores
            and the corresponding ngrams.
    """

    df = pd.DataFrame(columns=['Word', 'Keyness'])
    for ngram_len in range(1, max_ngram_len + 1):
        n_df = keyword_list_ngram(
            study_corpus,
            ref_corpus,
            metric,
            ngram_len,
            min_docs,
            min_freq,
            filter_stopwords,
            **kwargs
        )
        df = pd.concat([df, n_df], ignore_index=True)

    df = df.sort_values(by='Keyness', ascending=False)
    df = df.reset_index(drop=True)

    return df


def string_to_function(function_name: str) -> Callable:
    """Take a string and find the corresponding function in this module.

    Parameters:
        function_name (str): The name of the function to find.

    Returns:
        Callable: The function corresponding to the string.
    """
    # Get the current module
    current_module = sys.modules[__name__]

    # Check if the function name exists in the current module
    if hasattr(current_module, function_name):
        func = getattr(current_module, function_name)
        if callable(func):
            return func
        else:
            raise ValueError(
                f"{function_name} exists in the module but is not callable."
            )
    else:
        raise ValueError(f"Function {function_name} not found in module.")

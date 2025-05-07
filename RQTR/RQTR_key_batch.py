# Setup: Import necessary libraries

from src.corpus import Corpus, FrequencyCorpus
from src.metrics import keyness
from src.corpus_creation import document_retriever as dr
import pathlib
import json
import pandas as pd
import random
from src.load_data import load_files, load_reference_sample

# Put the path to the directory containing the corpus files here
CORPUSDIR = '/Users/django/Documents/GitHub.nosynch/Paper-ThematischesKorpus/final_corpus_annotated'

docs, metadata = load_files(CORPUSDIR)
corpus = Corpus(docs, metadata)

reference_corpus_scraped = FrequencyCorpus(docs, metadata)

#Parameter
methods = ["RQTR+OR"]
#methods = ["LL", "OR", "RQTR+LL", "RQTR+OR"]
search_terms = [['KI', ('künstlich', 'Intelligenz')], ['Roboter', ('künstlich', 'Intelligenz')], ['Chatbot', ('künstlich', 'Intelligenz')], ['Chatbot', 'Roboter', ('künstlich', 'Intelligenz')]]
min_hits = [1, 5]

rayson_threshold = 15.13
or_threshold = 1

### KEYNESS

def calculate_keyness(search_terms, metric): 

    # Find the documents that contain the search terms (at least min times)
    hits = dr.match_wordlist(
        corpus, search_terms, min=1
    )

    # Load the found documents into a new corpus
    study_corpus = dr.corpus_from_found(
        hits, source_corpus=corpus,
        goal_corpus=FrequencyCorpus
    )

    # create a keyword list for ngrams of length 1 and 2

    keynesses = keyness.keyword_list(
        study_corpus=study_corpus,
        ref_corpus=reference_corpus_scraped,
        metric=metric,
        max_ngram_len=2,
        min_docs=5,
        smoothing=0.0001,
        filter_stopwords=True,
    )

    # You can also save pandas dataframes (e.g. the keyword list) to a file
    keynesses.to_excel(f"{search_terms}_{metric}.xlsx", index=False)

    return keynesses

### RQTR CALCULATION

def rqtr_calculation(base_terms, keyness_metric):
    from copy import deepcopy
    corpus_copy = deepcopy(corpus)
    from src.metrics import rqtr_lemma

    baseline, core_term =rqtr_lemma.qtr_baseline(
        base_terms, corpus
    )

    cooccurence_values = rqtr_lemma.count_cooccurence(
        base_terms,
        corpus,
        max_ngram_len=2,
    )
    rqtrn_table = rqtr_lemma.cooccurence_to_metric(
        cooccurence_values,
        baseline,
        metric = 'rqtrn',
        min_docs=5,
    )

    # Again, you could save the rqtrn table to a file
    rqtrn_table.to_excel(f"{base_terms}_rqtrn_{keyness_metric}.xlsx", index=False)

    return rqtrn_table


### COMBINED

# Create a new dataframe with both keyness and rqtrn

def combine(df1, df2, base_terms):
    combined_df = pd.merge(
        df1,
        df2,
        on='Term',
        how='outer'
    )

    # Yet again, we can save it...
    combined_df.to_excel(f"{base_terms}_combined.xlsx", index=False)

    return combined_df


### FILTERING

def filter(df, values, thresholds, sort_by):
    # Drop rows that lack rqtrn or keyness values
    filtered_df = df.dropna()
    condition = " & ".join([f"(filtered_df['{v}'] > {t})" for (v,t) in zip(values, thresholds)])

    # Filter the dataframe to only include rows with rqtrn > 0 and keyness > 10
    filtered_df = filtered_df[
        eval(condition)
    ]

    # Keep only the top 50 rows after sorting by rqtrn
    filtered_df = filtered_df.sort_values(
        by=sort_by, ascending=False
    ).head(50)

    return filtered_df

# Let's take a look at the filtered dataframe
# filtered_df

def evaluate(filtered_df, min):
    found_docs = dr.match_wordlist(
        corpus,
        wordlist=filtered_df['Term'].tolist(),
        min=min  # Let's be strict

    )
    created_corpus = dr.corpus_from_found(
        found_docs,
        source_corpus=corpus,
        goal_corpus='Corpus'
    )

    dr.eval_retrieval(
        corpus,
        found_docs,
        annotator='gold_label',
        mode='pooling'
    )


for method in methods: 
    for search_term in search_terms:
        for min_hit in min_hits:
            print(f"Methode: {method}, search_terms: {search_term}, min_hits: {min_hit}")
            if method == "LL":
                threshold = rayson_threshold
                df=calculate_keyness(search_term, metric="log_likelihood_rayson")
                filtered_df = filter(df, values=["Keyness"], thresholds=[threshold], sort_by="Keyness")
                evaluate(filtered_df, min_hit)
            elif method == "OR":
                threshold = or_threshold
                df=calculate_keyness(search_term, metric="odds_ratio")
                filtered_df = filter(df, values=["Keyness"], thresholds=[threshold], sort_by="Keyness")
                evaluate(filtered_df, min_hit)
            elif method == "RQTR+LL":
                thresholds = [rayson_threshold, 0]
                df1=calculate_keyness(search_term, metric="log_likelihood_rayson")
                df2 = rqtr_calculation(search_term, keyness_metric="LL")
                combined_df = combine(df1, df2, search_term)
                filtered_df = filter(combined_df, values=["Keyness", "RQTRN"], thresholds=thresholds, sort_by="RQTRN")
                evaluate(filtered_df, min_hit)
            elif method == "RQTR+OR":
                thresholds = [or_threshold, 0]
                df1=calculate_keyness(search_term, metric="odds_ratio")
                df2 = rqtr_calculation(search_term, keyness_metric="OR")
                combined_df = combine(df1, df2, search_term)
                filtered_df = filter(combined_df, values=["Keyness", "RQTRN"], thresholds=thresholds, sort_by="RQTRN")
                evaluate(filtered_df, min_hit)



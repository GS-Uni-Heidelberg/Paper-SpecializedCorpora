# Evaluating Different Methods for Building Specialized Corpora: A Case Study on the German Discourse on AI

This repository contains the code used in our paper, enabling others to replicate our results. The data is only partially available for copyright reasons, but it can be reproduced by scraping the provided list of URLs.

The code can, of course, also be run on other data to apply the corpus-building methods from our paper to different contexts and corpus-building endeavors.

## Installation / Setup

The code was tested with Python 3.12. It is not guaranteed to work with other Python versions.

To install the necessary packages, run:

```bash
pip install -r requirements.txt
```

You must also download some additional data for the NLTK package and spaCy. To do this, run:

```bash
# Example for NLTK
python -m nltk.downloader all

# Example for spaCy (replace 'SPACY_PACKAGE' with the actual model, e.g., 'de_core_news_lg')
python -m spacy download SPACY_PACKAGE
```

If you want to use the Claude and/or OpenAI API, set up a `.env` file with your API keys.

If you want to run the code in the `rating/` directory, you must also install an additional Python module. See the README in that directory for details.

## Contents

The code is organized into the following four directories:

- **data** – Contains a table with all the articles/URLs used, plus metadata and ratings.
- **rating** – Code to handle the rating/categorization part of our paper, e.g., generating gold labels.
- **Metrics** – Implements the corpus linguistic and subjective metrics. Also contains code for the boolean search to be used with the wordlists those methods generate.
- **LLM** – Code to let LLMs rate articles or generate queries.

## Usage

Most usage should be straightforward or is explained in the respective notebooks/modules.

**Important:**
The code assumes that the data is in the following format:

```
dir/
  article1.json
  article2.json
  ...
```

And that the JSON files have the following schema:

```json
{
    "metadata1": "metadata",
    "metadata2": "more_metadata",
    ...
    "lemmas": [
        "lemma1",
        "lemma2",
        "lemma3"
    ],  // We used spaCy for lemmatization.
    "gold_label": "<1_hauptthema/2_nebenthema/3_kein_thema>"
}
```

Of course, you can use another format; you just need to adjust the code that loads the data accordingly.


# Citation

Please cite our paper if you use this code:

```bib
TBD
```
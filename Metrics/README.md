# Experimenting with Query Term Creation Methods
This directory contains code for testing different methods of creating query terms for specialized / thematic corpus creation.

## Setup
Install the required packages:
```bash
pip install -r requirements.txt
```

If you want to create an annotation folder / directory, for example to copy non-annotated documents into, you need to run the bash script `create_annotation_dir.sh`:
```bash
bash create_annotation_dir.sh <name_of_annotation_dir>
```


## Contents
This directory contains notebooks to test different methods and metrics. They are:
- **collocations.ipynb**: Classic corpus linguistics methods for identifying collocations in a corpus, used here to identify query terms.
- **keywords.ipynb**: Classic corpus linguistics methods for identifying keywords in a corpus, used here to identify query terms.
- **rqtr.ipynb**: QTR (Query Term Relevance) method for identifying query terms in a corpus, introduced by Gabrielatos.
- **rqtr_key.ipynb**: Combines RQTR with keyword extraction methods to identify query terms in a corpus.

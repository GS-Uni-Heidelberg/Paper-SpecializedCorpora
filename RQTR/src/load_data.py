from pathlib import Path
import json


def remove_keys(dict_, keys):
    """Remove keys from a dictionary."""
    for key in keys:
        if key in dict_:
            del dict_[key]
    return dict_


def get_source(url):
    if 'www.informatik-aktuell.de' in url:
        return 'infoakt'
    elif 'www.zeit.de' in url:
        return 'zeit'
    elif 'www.spektrum.de' in url:
        return 'spektrum'
    return 'other'


def load_files(
    dirpath, non_metadata_keys='lemmas'
):
    docs = []
    metadata = []
    for file in Path(dirpath).iterdir():
        with open(file, 'r') as f:
            doc = json.load(f)
            docs.append(doc['lemmas'])
            metadata_dict = remove_keys(doc, non_metadata_keys)

            if 'file' in metadata_dict:
                raise ValueError(
                    "The key 'file' is a reserved key "
                    "and cannot be used in the metadata dictionary."
                )
            metadata_dict['file'] = file.name
            metadata_dict['source'] = get_source(metadata_dict['url'])

            metadata.append(metadata_dict)

    return docs, metadata

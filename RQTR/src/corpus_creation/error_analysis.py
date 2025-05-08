from ..corpus import Corpus, FrequencyCorpus
from pathlib import Path


def keep_keys(
    dict_: dict,
    keys: list[str]
):
    """Function to keep only the keys in a dictionary."""
    return {
        key: dict_[key]
        for key in keys
        if key in dict_
    }


def false_positives(
    corpus: Corpus,
    found_docs: dict[list[str]],
    goal_path: str | Path,
    annotator: str = "gold_label",
    pos_labels: list[str] = ["1_hauptthema"],
):
    if isinstance(goal_path, str):
        goal_path = Path(goal_path)
    if not goal_path.exists():
        raise FileNotFoundError(f"Dir {goal_path} does not exist.")

    files_to_write = {}
    for i, entry in enumerate(corpus):
        _, metadata = entry
        if metadata.get(annotator, False) in pos_labels:
            continue  # True positive
        if i not in found_docs:
            continue  # Negative

        relevant_data = keep_keys(
            metadata,
            ['h1', 'url', 'text', 'og:description', 'source', 'file']
        )

        goal_text = (
            f"URL: {relevant_data['url']}\n"
            f"Title: {relevant_data['h1']}\n"
            f"Description: {relevant_data['og:description']}\n"
            f"Search terms: {', '.join(list(found_docs[i]))}\n\n"
            f"Label: {metadata.get(annotator, '3_kein_thema')}\n"
            f'++++++++++++++++++++++++++++++\n\n'
            f"{relevant_data['text']}\n"
        )

        filename = f"{Path(relevant_data['file']).stem}.txt"
        filepath = goal_path / filename
        files_to_write[filepath] = goal_text

    for filename, goal_text in files_to_write.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(goal_text)


def false_negatives(
    corpus: Corpus,
    found_docs: dict[list[str]],
    goal_path: str | Path,
    annotator: str = "gold_label",
    pos_labels: list[str] = ["1_hauptthema"],
):
    if isinstance(goal_path, str):
        goal_path = Path(goal_path)
    if not goal_path.exists():
        raise FileNotFoundError(f"Dir {goal_path} does not exist.")

    files_to_write = {}
    for i, entry in enumerate(corpus):
        _, metadata = entry
        if i in found_docs:
            continue  # Positives
        if metadata.get(annotator, False) not in pos_labels:
            continue  # True negatives

        relevant_data = keep_keys(
            metadata,
            ['h1', 'url', 'text', 'og:description', 'source', 'file']
        )

        goal_text = (
            f"URL: {relevant_data['url']}\n"
            f"Title: {relevant_data['h1']}\n"
            f"Description: {relevant_data['og:description']}\n"
            f"Search terms: {', '.join(list(found_docs.get(i, [])))}\n\n"
            f"Label: {metadata.get(annotator, '3_kein_thema')}\n"
            f'++++++++++++++++++++++++++++++\n\n'
            f"{relevant_data['text']}\n"
        )

        filename = f"{Path(relevant_data['file']).stem}.txt"
        filepath = goal_path / filename
        files_to_write[filepath] = goal_text

    for filename, goal_text in files_to_write.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(goal_text)

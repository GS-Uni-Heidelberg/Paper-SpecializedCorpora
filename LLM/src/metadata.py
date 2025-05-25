import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass
class RunMetadata:
    """Dataclass for storing metadata about a run."""
    run_id: str
    model: str
    time: str
    prompt_name: str
    prompt: str
    output_format: str
    dataset: str
    seed: int
    temperature: float


def flatten_report(report: dict) -> dict:
    """Takes an sklearn classification report
    and flattens it into a dictionary.
    """
    flat_report = {}
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):  # Skip non-dictionary items like acc
            for metric_name, value in metrics.items():
                flat_key = f"{class_name}_{metric_name}"
                flat_report[flat_key] = value
        else:
            flat_report[class_name] = metrics
    return flat_report


def append_to_runs(
    filepath: str | Path,
    run_metadata: RunMetadata | dict,
    classification_report: dict,
):
    """Append a new row to the `runs.tsv` file.

    Args:
        filepath (str | Path): Path to the `runs.csv` file.
        run_metadata (RunMetadata | dict): Metadata for the run.
        classification_report (dict): Classification report for the run.
    """

    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    if not isinstance(run_metadata, dict):
        run_metadata = run_metadata.__dict__

    flat_report = flatten_report(classification_report)
    combined_data = {**run_metadata, **flat_report}
    new_row = pd.DataFrame([combined_data])

    if Path(filepath).exists():
        runs = pd.read_csv(filepath, sep='\t')
    else:
        runs = pd.DataFrame()

    runs = pd.concat([runs, new_row], ignore_index=True)

    runs.to_csv(filepath, index=False, sep='\t')

"""Validity analysis for the complete_wcst dataset."""

from publication.preprocessing.constants import ANALYSIS_OUTPUT_DIR, get_results_dir
from publication.validity_reliability._core import validity

DATASET = "complete_wcst"
TASK = "wcst"
DATA_DIR = get_results_dir(TASK)
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "validity_reliability" / DATASET


def run():
    return validity.run(
        task=TASK,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    run()

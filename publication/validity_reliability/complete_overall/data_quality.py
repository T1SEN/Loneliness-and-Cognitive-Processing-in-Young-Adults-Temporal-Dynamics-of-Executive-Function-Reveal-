"""Data quality analysis for the complete_overall dataset."""

from publication.preprocessing.constants import ANALYSIS_OUTPUT_DIR, get_results_dir
from publication.validity_reliability._core import data_quality

DATASET = "complete_overall"
TASK = "overall"
TASKS = ["stroop", "wcst", "prp"]
DATA_DIR = get_results_dir(TASK)
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "validity_reliability" / DATASET


def run():
    return data_quality.run(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        tasks=TASKS,
    )


if __name__ == "__main__":
    run()

"""
Shared constants for data preprocessing.

Extracted from data_loader_utils.py for cleaner organization.
"""

from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = REPO_DIR / "outputs"
OUTPUT_STATS_DIR = OUTPUTS_DIR / "stats"
OUTPUT_TABLES_DIR = OUTPUTS_DIR / "tables"
OUTPUT_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUT_LOGS_DIR = OUTPUTS_DIR / "logs"
OUTPUT_STATS_CORE_DIR = OUTPUT_STATS_DIR / "core"
OUTPUT_STATS_SUPP_DIR = OUTPUT_STATS_DIR / "supplementary"
OUTPUT_TABLES_CORE_DIR = OUTPUT_TABLES_DIR / "core"
OUTPUT_TABLES_SUPP_DIR = OUTPUT_TABLES_DIR / "supplementary"
OUTPUT_FIGURES_CORE_DIR = OUTPUT_FIGURES_DIR / "core"
OUTPUT_FIGURES_SUPP_DIR = OUTPUT_FIGURES_DIR / "supplementary"
OUTPUT_LOGS_PIPELINE_DIR = OUTPUT_LOGS_DIR / "pipeline"
OUTPUT_LOGS_QC_DIR = OUTPUT_LOGS_DIR / "qc"
ANALYSIS_OUTPUT_DIR = OUTPUT_STATS_CORE_DIR

# Task-specific complete directories
COMPLETE_STROOP_DIR = DATA_DIR / "complete_stroop"
COMPLETE_WCST_DIR = DATA_DIR / "complete_wcst"
COMPLETE_OVERALL_DIR = DATA_DIR / "complete_overall"

# Overall-only mode
TASK_DIRS = {
    'overall': COMPLETE_OVERALL_DIR,
}

VALID_TASKS = {'overall'}


def get_results_dir(task: str) -> Path:
    """Return the task-specific data directory.

    Args:
        task: 'stroop', 'wcst', or 'overall'

    Returns:
        Path to the task-specific complete directory
    """
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {VALID_TASKS}")
    return TASK_DIRS[task]

# RT filtering constants
DEFAULT_RT_MIN = 100          # ms; drop anticipations
DEFAULT_RT_MAX = 5000         # ms; general upper bound (WCST etc.)
STROOP_RT_MAX = 3000          # ms; Stroop task timeout is 3s
STROOP_RT_MIN = 200           # ms; Stroop RT lower bound for trial-level QC

# WCST filtering constants
WCST_RT_MIN = 200             # ms; anticipatory response cutoff
WCST_RT_MAX = 10000           # ms; upper bound for post-error RT filtering
WCST_VALID_CONDS = {"colour", "shape", "number"}
WCST_VALID_CARDS = {
    "one_yellow_circle",
    "two_black_rectangle",
    "three_blue_star",
    "four_red_triangle",
}
WCST_MIN_TRIALS = 60          # theoretical minimum
WCST_MIN_MEDIAN_RT = 300      # ms; random clicking threshold
WCST_MAX_SINGLE_CHOICE = 0.85 # max ratio for single card choice

# Columns to standardize
STANDARDIZE_COLS = ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']

# Gender normalization tokens
# Gender normalization tokens
MALE_TOKENS_EXACT = {"m", "male", "man", "men", "boy", "boys", "\uB0A8", "\uB0A8\uC131", "\uB0A8\uC790"}
FEMALE_TOKENS_EXACT = {"f", "female", "woman", "women", "girl", "girls", "\uC5EC", "\uC5EC\uC131", "\uC5EC\uC790"}
MALE_TOKENS_CONTAINS = {"\uB0A8\uC131", "\uB0A8\uC790"}
FEMALE_TOKENS_CONTAINS = {"\uC5EC\uC131", "\uC5EC\uC790"}


# Participant ID aliases
PARTICIPANT_ID_ALIASES = {"participant_id", "participantId", "participantid"}

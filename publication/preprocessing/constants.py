"""
Shared constants for data preprocessing.

Extracted from data_loader_utils.py for cleaner organization.
"""

from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ANALYSIS_OUTPUT_DIR = DATA_DIR / "outputs"

# Task-specific complete directories
COMPLETE_STROOP_DIR = DATA_DIR / "complete_stroop"
COMPLETE_PRP_DIR = DATA_DIR / "complete_prp"
COMPLETE_WCST_DIR = DATA_DIR / "complete_wcst"
COMPLETE_OVERALL_DIR = DATA_DIR / "complete_overall"

TASK_DIRS = {
    'stroop': COMPLETE_STROOP_DIR,
    'prp': COMPLETE_PRP_DIR,
    'wcst': COMPLETE_WCST_DIR,
    'overall': COMPLETE_OVERALL_DIR,
}

VALID_TASKS = {'stroop', 'prp', 'wcst', 'overall'}


def get_results_dir(task: str) -> Path:
    """Return the task-specific data directory.

    Args:
        task: 'stroop', 'prp', 'wcst', or 'overall'

    Returns:
        Path to the task-specific complete directory
    """
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {VALID_TASKS}")
    return TASK_DIRS[task]

# RT filtering constants
DEFAULT_RT_MIN = 100          # ms; drop anticipations
DEFAULT_RT_MAX = 5000         # ms; general upper bound (WCST etc.)
PRP_RT_MAX = 3000             # ms; PRP task timeout is 3s
PRP_RT_MIN = 200              # ms; PRP RT2 lower bound for anticipation
STROOP_RT_MAX = 3000          # ms; Stroop task timeout is 3s
STROOP_RT_MIN = 200           # ms; Stroop RT lower bound for trial-level QC

# PRP SOA binning constants
DEFAULT_SOA_SHORT = 150       # ms; short bin upper bound
DEFAULT_SOA_LONG = 1200       # ms; long bin lower bound

# PRP QC constants
PRP_IRI_MIN = 100             # ms; response grouping threshold
PRP_ACC_THRESHOLD = 0.70      # participant-level accuracy cutoff
PRP_LONG_SOA_MIN = DEFAULT_SOA_LONG  # ms; long SOA definition for QC (aligned with DEFAULT_SOA_LONG)

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
MALE_TOKENS_EXACT = {"m", "male", "man", "men", "boy", "boys"}
FEMALE_TOKENS_EXACT = {"f", "female", "woman", "women", "girl", "girls"}
MALE_TOKENS_CONTAINS = {"남성", "남자", "소년", "남학생"}
FEMALE_TOKENS_CONTAINS = {"여성", "여자", "소녀", "여학생"}

# Participant ID aliases
PARTICIPANT_ID_ALIASES = {"participant_id", "participantId", "participantid"}

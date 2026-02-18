"""Shared constants for preprocessing and analysis (public-only runtime)."""

from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "data"
RAW_DIR = DATA_DIR / "raw"  # retained for internal public-data generation
PUBLIC_DIR = DATA_DIR / "public"

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

# Public bundle files (official runtime inputs)
PUBLIC_FILE_MAP = {
    "demographics": "demographics_public.csv",
    "surveys": "surveys_public.csv",
    "features": "features_public.csv",
    "stroop_trials": "stroop_trials_public.csv",
    "wcst_trials": "wcst_trials_public.csv",
}

# Overall-only mode
TASK_DIRS = {
    "overall": PUBLIC_DIR,
}
VALID_TASKS = {"overall"}


def get_results_dir(task: str) -> Path:
    """Return task-specific runtime data directory (public-only)."""
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {VALID_TASKS}")
    return TASK_DIRS[task]


def get_public_file(key: str) -> Path:
    """Return required public bundle file path by logical key."""
    if key not in PUBLIC_FILE_MAP:
        raise ValueError(f"Unknown public file key: {key}. Valid keys: {sorted(PUBLIC_FILE_MAP)}")
    return PUBLIC_DIR / PUBLIC_FILE_MAP[key]


def get_stroop_trials_path(task: str = "overall") -> Path:
    """Return Stroop trials CSV path for task."""
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {VALID_TASKS}")
    return get_public_file("stroop_trials")


def get_wcst_trials_path(task: str = "overall") -> Path:
    """Return WCST trials CSV path for task."""
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {VALID_TASKS}")
    return get_public_file("wcst_trials")


# RT filtering constants
DEFAULT_RT_MIN = 100
DEFAULT_RT_MAX = 5000
STROOP_RT_MAX = 3000
STROOP_RT_MIN = 200

# WCST filtering constants
WCST_RT_MIN = 200
WCST_RT_MAX = 10000
WCST_VALID_CONDS = {"colour", "shape", "number"}
WCST_VALID_CARDS = {
    "one_yellow_circle",
    "two_black_rectangle",
    "three_blue_star",
    "four_red_triangle",
}
WCST_MIN_TRIALS = 60
WCST_MIN_MEDIAN_RT = 300
WCST_MAX_SINGLE_CHOICE = 0.85

# Columns to standardize
STANDARDIZE_COLS = ["ucla_score", "dass_depression", "dass_anxiety", "dass_stress", "age"]

# Gender normalization tokens
MALE_TOKENS_EXACT = {"m", "male", "man", "men", "boy", "boys", "\uB0A8", "\uB0A8\uC131", "\uB0A8\uC790"}
FEMALE_TOKENS_EXACT = {"f", "female", "woman", "women", "girl", "girls", "\uC5EC", "\uC5EC\uC131", "\uC5EC\uC790"}
MALE_TOKENS_CONTAINS = {"\uB0A8\uC131", "\uB0A8\uC790"}
FEMALE_TOKENS_CONTAINS = {"\uC5EC\uC131", "\uC5EC\uC790"}

# Participant ID aliases
PARTICIPANT_ID_ALIASES = {"participant_id", "participantId", "participantid", "public_id"}

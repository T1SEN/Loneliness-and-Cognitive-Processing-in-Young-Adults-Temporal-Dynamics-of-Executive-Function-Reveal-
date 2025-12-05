"""
Shared constants for data preprocessing.

Extracted from data_loader_utils.py for cleaner organization.
"""

from pathlib import Path

# Directory paths
RESULTS_DIR = Path("results/complete_only")
ANALYSIS_OUTPUT_DIR = Path("results/analysis_outputs")

# RT filtering constants
DEFAULT_RT_MIN = 100          # ms; drop anticipations
DEFAULT_RT_MAX = 5000         # ms; legacy upper bound (WCST etc.)
PRP_RT_MAX = 3000             # ms; PRP task timeout is 3s
STROOP_RT_MAX = 3000          # ms; Stroop task timeout is 3s

# PRP SOA binning constants
DEFAULT_SOA_SHORT = 150       # ms; short bin upper bound
DEFAULT_SOA_LONG = 1200       # ms; long bin lower bound

# Cache path for master dataset
MASTER_CACHE_PATH = ANALYSIS_OUTPUT_DIR / "master_dataset.parquet"

# Columns to standardize
STANDARDIZE_COLS = ['ucla_score', 'dass_depression', 'dass_anxiety', 'dass_stress', 'age']

# Gender normalization tokens
MALE_TOKENS_EXACT = {"m", "male", "man", "men", "boy", "boys"}
FEMALE_TOKENS_EXACT = {"f", "female", "woman", "women", "girl", "girls"}
MALE_TOKENS_CONTAINS = {"남성", "남자", "소년", "남학생"}
FEMALE_TOKENS_CONTAINS = {"여성", "여자", "소녀", "여학생"}

# Participant ID aliases
PARTICIPANT_ID_ALIASES = {"participant_id", "participantId", "participantid"}

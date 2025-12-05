"""
Cross-Task Suite - Common Utilities
===================================

Shared code for cross-task analysis modules.
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import statsmodels.formula.api as smf

# Project imports
from analysis.preprocessing import (
    load_master_dataset,
    ANALYSIS_OUTPUT_DIR,
    safe_zscore,
    prepare_gender_variable,
    find_interaction_term,
)
from analysis.utils.modeling import standardize_predictors

np.random.seed(42)

# Output directory
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "cross_task_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ANALYSIS REGISTRY
# =============================================================================

@dataclass
class AnalysisSpec:
    """Specification for an analysis."""
    name: str
    description: str
    function: Callable
    source_script: str


def register_analysis(registry: Dict[str, AnalysisSpec], name: str, description: str, source_script: str):
    """Decorator to register an analysis function to a specific registry."""
    def decorator(func: Callable):
        registry[name] = AnalysisSpec(
            name=name,
            description=description,
            function=func,
            source_script=source_script
        )
        return func
    return decorator


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cross_task_data() -> pd.DataFrame:
    """Load and prepare master dataset for cross-task analyses."""
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    # Normalize gender using shared utility
    master = prepare_gender_variable(master)

    if 'ucla_total' not in master.columns and 'ucla_score' in master.columns:
        master['ucla_total'] = master['ucla_score']

    # Standardize predictors
    master = standardize_predictors(master)

    return master


# =============================================================================
# UTILITIES
# =============================================================================

def residualize_for_covariates(
    df: pd.DataFrame,
    target_cols: List[str],
    covars: List[str]
) -> pd.DataFrame:
    """
    Regress out covariates from each target column and return residuals.
    """
    resid_df = pd.DataFrame(index=df.index)
    for col in target_cols:
        cols_needed = [col] + covars
        sub = df.dropna(subset=cols_needed)
        if len(sub) < 10:
            continue
        formula = f"{col} ~ " + " + ".join(covars)
        model = smf.ols(formula, data=sub).fit()
        resid_df.loc[sub.index, f"{col}_resid"] = model.resid
    return resid_df

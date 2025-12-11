"""
Network Analysis Configuration
==============================

Defines reusable variable sets and paths for publication-grade network analyses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from publication.preprocessing.constants import ANALYSIS_OUTPUT_DIR


# =============================================================================
# OUTPUT PATH
# =============================================================================

BASE_OUTPUT = ANALYSIS_OUTPUT_DIR / "network_analysis"
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# VARIABLE SET DEFINITIONS
# =============================================================================

@dataclass
class NetworkVariableSet:
    """
    Container describing which variables enter a network model.

    Attributes
    ----------
    name : str
        Identifier used for folder names and CLI arguments.
    columns : list of str
        Required columns. If missing from the dataset, the network will abort.
    description : str
        Short human-readable summary.
    labels : dict
        Mapping from column name to display label.
    communities : dict
        Mapping from column name to community/category (e.g., "Affect", "EF").
    optional_columns : list of str, optional
        Additional columns to include when available.
    min_n : int
        Minimum number of rows required to estimate the network.
    """

    name: str
    columns: List[str]
    description: str
    labels: Dict[str, str]
    communities: Dict[str, str]
    optional_columns: Optional[List[str]] = field(default=None)
    aliases: Dict[str, List[str]] = field(default_factory=dict)
    min_n: int = 120

    def available_columns(self, df_columns: Iterable[str]) -> List[str]:
        """Return columns present in the supplied dataframe."""
        df_columns = set(df_columns)
        base = [col for col in self.columns if col in df_columns]
        optional = [
            col for col in (self.optional_columns or []) if col in df_columns
        ]
        return base + optional

    def get_label(self, column: str) -> str:
        """Return a human-readable label for a column."""
        return self.labels.get(column, column)

    def get_community(self, column: str) -> str:
        """Return a community label for a column."""
        return self.communities.get(column, "unspecified")

    def apply_aliases(self, df):
        """Add canonical columns by copying from aliases when necessary."""
        if not self.aliases:
            return df
        result = df.copy()
        for canonical, candidates in self.aliases.items():
            if canonical in result.columns:
                continue
            for alias in candidates:
                if alias in result.columns:
                    result[canonical] = result[alias]
                    break
        return result


DOMAIN_LEVEL_SET = NetworkVariableSet(
    name="domain_level",
    description=(
        "Loneliness + DASS (Dep/Anx/Str) + canonical EF metrics "
        "(WCST PE, Stroop interference, PRP bottleneck)."
    ),
    columns=[
        "z_ucla",
        "z_dass_dep",
        "z_dass_anx",
        "z_dass_str",
        "pe_rate",
        "stroop_interference",
        "prp_bottleneck",
    ],
    aliases={
        "z_dass_dep": ["z_dass_depression"],
        "z_dass_anx": ["z_dass_anxiety"],
        "z_dass_str": ["z_dass_stress"],
    },
    optional_columns=[
        # Mechanism-level extensions (only used if present)
        "prp_sigma_short",
        "prp_sigma_long",
        "prp_sigma_bottleneck",
        "prp_tau_short",
        "prp_tau_long",
        "hmm_lapse_occupancy",
        "lapse_occupancy",
        "trans_to_lapse",
        "trans_to_focus",
        "wcst_post_error_slowing",
        "prp_post_error_slowing",
        "stroop_rt_cv_all",
    ],
    labels={
        "z_ucla": "UCLA Loneliness",
        "z_dass_dep": "DASS Depression",
        "z_dass_anx": "DASS Anxiety",
        "z_dass_str": "DASS Stress",
        "pe_rate": "WCST PE Rate",
        "stroop_interference": "Stroop Interference",
        "prp_bottleneck": "PRP Bottleneck (ms)",
        "prp_sigma_short": "PRP σ (Short SOA)",
        "prp_sigma_long": "PRP σ (Long SOA)",
        "prp_sigma_bottleneck": "PRP σ Difference",
        "prp_tau_short": "PRP τ (Short SOA)",
        "prp_tau_long": "PRP τ (Long SOA)",
        "hmm_lapse_occupancy": "WCST Lapse Occupancy",
        "lapse_occupancy": "Lapse Occupancy",
        "trans_to_lapse": "P(Focus→Lapse)",
        "trans_to_focus": "P(Lapse→Focus)",
        "wcst_post_error_slowing": "WCST PES",
        "prp_post_error_slowing": "PRP PES",
        "stroop_rt_cv_all": "Stroop RT CV",
    },
    communities={
        "z_ucla": "Affect",
        "z_dass_dep": "Affect",
        "z_dass_anx": "Affect",
        "z_dass_str": "Affect",
        "pe_rate": "Executive",
        "stroop_interference": "Executive",
        "prp_bottleneck": "Executive",
        "prp_sigma_short": "Mechanism",
        "prp_sigma_long": "Mechanism",
        "prp_sigma_bottleneck": "Mechanism",
        "prp_tau_short": "Mechanism",
        "prp_tau_long": "Mechanism",
        "hmm_lapse_occupancy": "Mechanism",
        "lapse_occupancy": "Mechanism",
        "trans_to_lapse": "Mechanism",
        "trans_to_focus": "Mechanism",
        "wcst_post_error_slowing": "Mechanism",
        "prp_post_error_slowing": "Mechanism",
        "stroop_rt_cv_all": "Mechanism",
    },
    min_n=60,
)


VARIABLE_SETS: Dict[str, NetworkVariableSet] = {
    DOMAIN_LEVEL_SET.name: DOMAIN_LEVEL_SET,
}


__all__ = [
    "BASE_OUTPUT",
    "NetworkVariableSet",
    "VARIABLE_SETS",
]

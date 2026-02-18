"""Survey QC helpers for public-only runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from ..constants import get_public_file
from ..core import ensure_participant_id, normalize_gender_series


@dataclass
class SurveyQCCriteria:
    ucla_score_required: bool = True
    dass_subscales_required: bool = True
    gender_required: bool = True
    date_filter: Optional[pd.Timestamp] = field(default=None)


def get_survey_valid_participants(
    data_dir: Optional[Path] = None,
    criteria: Optional[SurveyQCCriteria] = None,
    verbose: bool = False,
) -> Set[str]:
    _ = data_dir
    if criteria is None:
        criteria = SurveyQCCriteria()

    surveys_path = get_public_file("surveys")
    demo_path = get_public_file("demographics")

    if not surveys_path.exists() or not demo_path.exists():
        if verbose:
            print("[WARN] Missing required public survey/demographics files.")
        return set()

    surveys = pd.read_csv(surveys_path, encoding="utf-8-sig")
    surveys = ensure_participant_id(surveys)
    if "surveyName" not in surveys.columns:
        return set()

    surveys["surveyName"] = surveys["surveyName"].astype(str).str.lower()

    ucla_df = surveys[surveys["surveyName"] == "ucla"]
    if criteria.ucla_score_required:
        ucla_valid = set(ucla_df[pd.to_numeric(ucla_df.get("score"), errors="coerce").notna()]["participant_id"].astype(str))
    else:
        ucla_valid = set(ucla_df["participant_id"].dropna().astype(str))

    dass_df = surveys[surveys["surveyName"].str.contains("dass", na=False)]
    if criteria.dass_subscales_required and all(c in dass_df.columns for c in ["score_D", "score_A", "score_S"]):
        mask = (
            pd.to_numeric(dass_df["score_D"], errors="coerce").notna()
            & pd.to_numeric(dass_df["score_A"], errors="coerce").notna()
            & pd.to_numeric(dass_df["score_S"], errors="coerce").notna()
        )
        dass_valid = set(dass_df.loc[mask, "participant_id"].dropna().astype(str))
    else:
        dass_valid = set(dass_df["participant_id"].dropna().astype(str))

    valid_ids = ucla_valid & dass_valid

    if criteria.gender_required:
        demo = pd.read_csv(demo_path, encoding="utf-8-sig")
        demo = ensure_participant_id(demo)
        if "gender" in demo.columns:
            gender_norm = normalize_gender_series(demo["gender"])
            gender_valid = set(demo.loc[gender_norm.notna(), "participant_id"].dropna().astype(str))
            valid_ids = valid_ids & gender_valid

    if criteria.date_filter is not None and verbose:
        print("[INFO] date_filter is ignored in public-only mode (no timestamp fields).")

    if verbose:
        print(f"Survey QC valid participants: {len(valid_ids)}")

    return valid_ids

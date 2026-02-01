"""
Survey QC helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from ..constants import RAW_DIR


@dataclass
class SurveyQCCriteria:
    ucla_score_required: bool = True
    dass_subscales_required: bool = True
    gender_required: bool = True
    date_filter: Optional[pd.Timestamp] = field(
        default_factory=lambda: pd.Timestamp("2025-09-01", tz="UTC")
    )


def get_survey_valid_participants(
    data_dir: Optional[Path] = None,
    criteria: Optional[SurveyQCCriteria] = None,
    verbose: bool = False,
) -> Set[str]:
    if data_dir is None:
        data_dir = RAW_DIR
    if criteria is None:
        criteria = SurveyQCCriteria()

    valid_ids: Set[str] = set()

    surveys_path = data_dir / "2_surveys_results.csv"
    if not surveys_path.exists():
        if verbose:
            print(f"[WARN] surveys file not found: {surveys_path}")
        return valid_ids

    surveys = pd.read_csv(surveys_path, encoding="utf-8-sig")
    surveys["surveyName"] = surveys["surveyName"].str.lower()

    ucla_df = surveys[surveys["surveyName"] == "ucla"]
    if criteria.ucla_score_required:
        ucla_valid = set(ucla_df[ucla_df["score"].notna()]["participantId"].unique())
    else:
        ucla_valid = set(ucla_df["participantId"].unique())

    dass_df = surveys[surveys["surveyName"].str.contains("dass", na=False)]
    if criteria.dass_subscales_required:
        dass_valid = set(
            dass_df[
                (dass_df["score_D"].notna())
                & (dass_df["score_A"].notna())
                & (dass_df["score_S"].notna())
            ]["participantId"].unique()
        )
    else:
        dass_valid = set(dass_df["participantId"].unique())

    valid_ids = ucla_valid & dass_valid

    if verbose:
        print(
            f"Survey complete: {len(valid_ids)} (UCLA: {len(ucla_valid)}, DASS: {len(dass_valid)})"
        )

    if criteria.gender_required:
        participants_path = data_dir / "1_participants_info.csv"
        if participants_path.exists():
            participants_df = pd.read_csv(participants_path, encoding="utf-8-sig")
            missing_gender_ids = set(
                participants_df[
                    participants_df["gender"].isna()
                    | (participants_df["gender"].astype(str).str.strip() == "")
                ]["participantId"].dropna()
            )
            to_remove = valid_ids & missing_gender_ids
            if to_remove and verbose:
                print(f"  [INFO] excluded for missing gender: {len(to_remove)}")
            valid_ids -= to_remove

    if criteria.date_filter is not None:
        participants_path = data_dir / "1_participants_info.csv"
        if participants_path.exists():
            participants_df = pd.read_csv(participants_path, encoding="utf-8-sig")
            if "createdAt" in participants_df.columns:
                participants_df["createdAt"] = pd.to_datetime(
                    participants_df["createdAt"], errors="coerce", utc=True
                )
                recent_ids = set(
                    participants_df[participants_df["createdAt"] >= criteria.date_filter][
                        "participantId"
                    ].dropna()
                )
                excluded = valid_ids - recent_ids
                if excluded and verbose:
                    print(f"  [INFO] excluded for date cutoff: {len(excluded)}")
                valid_ids = valid_ids & recent_ids

    return valid_ids

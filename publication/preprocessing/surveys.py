"""
Survey loaders and QC filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from .constants import RAW_DIR
from .core import ensure_participant_id, normalize_gender_series


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


def load_participants(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "1_participants_info.csv", encoding="utf-8")
    df = ensure_participant_id(df)
    df["gender"] = normalize_gender_series(df["gender"])
    return df[["participant_id", "age", "gender", "education"]]


def load_ucla_scores(data_dir: Path) -> pd.DataFrame:
    surveys = pd.read_csv(data_dir / "2_surveys_results.csv", encoding="utf-8")
    surveys = ensure_participant_id(surveys)

    if "surveyName" in surveys.columns:
        ucla_data = surveys[surveys["surveyName"].str.lower() == "ucla"].copy()
    else:
        raise KeyError("No survey name column found")

    ucla_data["score"] = pd.to_numeric(ucla_data["score"], errors="coerce")
    ucla_data = ucla_data.dropna(subset=["score"])
    ucla_data = ucla_data.drop_duplicates(subset=["participant_id"], keep="last")

    ucla_scores = ucla_data[["participant_id", "score"]].rename(columns={"score": "ucla_total"})
    ucla_scores.columns = ["participant_id", "ucla_total"]
    return ucla_scores


def load_dass_scores(data_dir: Path) -> pd.DataFrame:
    surveys = pd.read_csv(data_dir / "2_surveys_results.csv", encoding="utf-8")
    surveys = ensure_participant_id(surveys)

    if "surveyName" in surveys.columns:
        dass_data = surveys[surveys["surveyName"].str.lower().str.contains("dass")].copy()
    else:
        return pd.DataFrame(columns=["participant_id", "dass_depression", "dass_anxiety", "dass_stress"])

    if len(dass_data) == 0:
        return pd.DataFrame(columns=["participant_id", "dass_depression", "dass_anxiety", "dass_stress"])

    if all(col in dass_data.columns for col in ["score_D", "score_A", "score_S"]):
        for col in ["score_D", "score_A", "score_S"]:
            dass_data[col] = pd.to_numeric(dass_data[col], errors="coerce")

        sort_cols = ["participant_id"]
        if "createdAt" in dass_data.columns:
            sort_cols.append("createdAt")
        elif "created_at" in dass_data.columns:
            sort_cols.append("created_at")
        dass_data = dass_data.sort_values(sort_cols)

        dass_summary = dass_data.drop_duplicates(subset=["participant_id"], keep="last")[
            ["participant_id", "score_D", "score_A", "score_S"]
        ].copy()
        dass_summary.columns = ["participant_id", "dass_depression", "dass_anxiety", "dass_stress"]
    else:
        if "score" not in dass_data.columns:
            raise KeyError("DASS responses missing 'score' column; cannot compute component scores.")
        dass_data["score"] = pd.to_numeric(dass_data["score"], errors="coerce")
        dass_data = dass_data.dropna(subset=["score"])
        dass_scores = dass_data.groupby(["participant_id", "questionText"])["score"].sum().unstack(fill_value=0)

        dep_items = [col for col in dass_scores.columns if any(x in str(col).lower() for x in ["meaningless", "nothing", "enthused", "worth", "positive", "initiative", "future"])]
        anx_items = [col for col in dass_scores.columns if any(x in str(col).lower() for x in ["breathing", "trembling", "worried", "panic", "heart", "scared", "dry"])]
        stress_items = [col for col in dass_scores.columns if any(x in str(col).lower() for x in ["wind down", "over-react", "nervous", "agitated", "relax", "intolerant", "touchy"])]

        dass_summary = pd.DataFrame()
        dass_summary["participant_id"] = dass_scores.index
        dass_summary["dass_depression"] = dass_scores[dep_items].sum(axis=1).values if dep_items else 0
        dass_summary["dass_anxiety"] = dass_scores[anx_items].sum(axis=1).values if anx_items else 0
        dass_summary["dass_stress"] = dass_scores[stress_items].sum(axis=1).values if stress_items else 0
        dass_summary = dass_summary.reset_index(drop=True)

    return dass_summary


def load_survey_items(data_dir: Path) -> pd.DataFrame:
    surveys = pd.read_csv(data_dir / "2_surveys_results.csv", encoding="utf-8")
    surveys = ensure_participant_id(surveys)

    def _extract_items(df: pd.DataFrame, prefix: str, n_items: int) -> pd.DataFrame:
        cols = [f"q{i}" for i in range(1, n_items + 1) if f"q{i}" in df.columns]
        if not cols:
            return pd.DataFrame(columns=["participant_id"])
        tmp = df[["participant_id"] + cols].copy()
        tmp = tmp.groupby("participant_id").first().reset_index()
        rename_map = {c: f"{prefix}_{c.lstrip('q')}" for c in cols}
        tmp = tmp.rename(columns=rename_map)
        for c in rename_map.values():
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        return tmp

    ucla_df = pd.DataFrame(columns=["participant_id"])
    dass_df = pd.DataFrame(columns=["participant_id"])

    if "surveyName" in surveys.columns:
        ucla_df = surveys[surveys["surveyName"].str.lower() == "ucla"].copy()
        dass_df = surveys[surveys["surveyName"].str.lower().str.contains("dass")].copy()

    ucla_items = _extract_items(ucla_df, "ucla", 20)
    dass_items = _extract_items(dass_df, "dass", 21)

    survey_items = ucla_items.merge(dass_items, on="participant_id", how="outer")
    return survey_items

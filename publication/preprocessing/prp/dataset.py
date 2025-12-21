"""PRP dataset builder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

from ..constants import PRP_RT_MIN, RAW_DIR, get_results_dir
from ..surveys import get_survey_valid_participants, SurveyQCCriteria
from .filters import get_prp_valid_participants, PRPQCCriteria

TASK_FILES = [
    "1_participants_info.csv",
    "2_surveys_results.csv",
    "3_cognitive_tests_summary.csv",
    "4a_prp_trials.csv",
]


def get_prp_complete_participants(
    data_dir: Optional[Path] = None,
    survey_criteria: Optional[SurveyQCCriteria] = None,
    task_criteria: Optional[PRPQCCriteria] = None,
    verbose: bool = False,
) -> Set[str]:
    if data_dir is None:
        data_dir = RAW_DIR

    survey_valid = get_survey_valid_participants(data_dir, survey_criteria, verbose)
    prp_valid = get_prp_valid_participants(data_dir, task_criteria, verbose)
    return survey_valid & prp_valid


def build_prp_dataset(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    survey_criteria: Optional[SurveyQCCriteria] = None,
    task_criteria: Optional[PRPQCCriteria] = None,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    if data_dir is None:
        data_dir = RAW_DIR
    if output_dir is None:
        output_dir = get_results_dir("prp")

    if verbose:
        print("=" * 60)
        print("PRP dataset build")
        print("=" * 60)

    valid_ids = get_prp_complete_participants(
        data_dir=data_dir,
        survey_criteria=survey_criteria,
        task_criteria=task_criteria,
        verbose=verbose,
    )

    if not valid_ids:
        if verbose:
            print("[WARN] no valid PRP participants")
        return {}

    if verbose:
        print(f"\nValid participants: {len(valid_ids)}")

    if save:
        os.makedirs(output_dir, exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}

    for filename in TASK_FILES:
        input_path = data_dir / filename
        if not input_path.exists():
            if verbose:
                print(f"  [SKIP] {filename} not found")
            continue

        df = pd.read_csv(input_path, encoding="utf-8-sig")
        original_count = len(df)

        if "participantId" not in df.columns:
            if verbose:
                print(f"  [ERROR] {filename} missing participantId")
            continue

        df_filtered = df[df["participantId"].isin(valid_ids)].copy()
        if filename == "3_cognitive_tests_summary.csv" and "testName" in df_filtered.columns:
            df_filtered["testName"] = df_filtered["testName"].str.lower()
            df_filtered = df_filtered[df_filtered["testName"] == "prp"]
        if filename == "4a_prp_trials.csv":
            rt_col = "t2_rt_ms" if "t2_rt_ms" in df_filtered.columns else "t2_rt" if "t2_rt" in df_filtered.columns else None
            if rt_col is None:
                if verbose:
                    print("  [ERROR] PRP trials missing T2 RT column")
                continue
            df_filtered[rt_col] = pd.to_numeric(df_filtered[rt_col], errors="coerce")

            for timeout_col in ("t1_timeout", "t2_timeout"):
                if timeout_col in df_filtered.columns:
                    timeout = df_filtered[timeout_col]
                    if timeout.dtype != bool:
                        timeout = timeout.astype(str).str.strip().str.lower().map(
                            {"true": True, "1": True, "false": False, "0": False}
                        )
                        timeout = timeout.fillna(False)
                    df_filtered[timeout_col] = timeout.astype(bool)
                    df_filtered = df_filtered[df_filtered[timeout_col] == False]

            if "response_order" in df_filtered.columns:
                def _norm_order(value: object) -> str | None:
                    if not isinstance(value, str):
                        return None
                    cleaned = value.strip().lower()
                    if not cleaned:
                        return None
                    token = "".join(ch for ch in cleaned if ch.isalnum())
                    if token.startswith("t1t2"):
                        return "t1_t2"
                    if token.startswith("t2t1"):
                        return "t2_t1"
                    if token in {"t1only", "t2only", "none"}:
                        return token
                    return None

                order_norm = df_filtered["response_order"].apply(_norm_order)
                df_filtered = df_filtered[order_norm == "t1_t2"]
            if "t2_pressed_while_t1_pending" in df_filtered.columns:
                pending = df_filtered["t2_pressed_while_t1_pending"]
                if pending.dtype != bool:
                    pending = pending.astype(str).str.strip().str.lower().map(
                        {"true": True, "1": True, "false": False, "0": False}
                    )
                    pending = pending.fillna(False)
                df_filtered["t2_pressed_while_t1_pending"] = pending.astype(bool)
                df_filtered = df_filtered[df_filtered["t2_pressed_while_t1_pending"] == False]

            df_filtered = df_filtered[df_filtered[rt_col].notna()]
            df_filtered = df_filtered[df_filtered[rt_col] >= PRP_RT_MIN]

        results[filename] = df_filtered

        if save:
            output_path = output_dir / filename
            df_filtered.to_csv(output_path, index=False, encoding="utf-8-sig")

        if verbose:
            print(f"  [OK] {filename}: {original_count} -> {len(df_filtered)} rows")

    if verbose:
        print(f"\nDone: '{output_dir}'")

    if save:
        ids_path = output_dir / "filtered_participant_ids.csv"
        pd.DataFrame({"participantId": sorted(valid_ids)}).to_csv(ids_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"  [OK] participant ids: {ids_path}")

    return results

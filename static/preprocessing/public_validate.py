"""
Validation utilities for the public-only data bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .constants import PUBLIC_FILE_MAP, get_public_file
from .core import normalize_gender_series


REQUIRED_COLUMNS: dict[str, set[str]] = {
    "demographics": {"public_id", "gender", "age"},
    "surveys": {"public_id", "surveyName"},
    "features": {
        "public_id",
        "stroop_interference",
        "stroop_interference_slope",
        "wcst_perseverative_error_rate",
        "wcst_exploration_rt_all",
        "wcst_confirmation_rt_all",
        "wcst_exploitation_rt_all",
        "wcst_confirmation_minus_exploitation_rt_all",
    },
    "stroop_trials": {"public_id"},
    "wcst_trials": {"public_id"},
}


@dataclass
class PublicValidationResult:
    ok: bool
    n_common_ids: int
    n_union_ids: int
    issues: list[str]


def _read_public_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def load_public_frames() -> dict[str, pd.DataFrame]:
    return {key: _read_public_csv(get_public_file(key)) for key in PUBLIC_FILE_MAP}


def _id_set(df: pd.DataFrame) -> set[str]:
    if "public_id" not in df.columns:
        return set()
    return set(df["public_id"].dropna().astype(str))


def validate_public_bundle(raise_on_error: bool = True) -> PublicValidationResult:
    issues: list[str] = []
    frames = load_public_frames()

    # File/column checks
    for key, filename in PUBLIC_FILE_MAP.items():
        path = get_public_file(key)
        if not path.exists():
            issues.append(f"Missing required file: {filename}")
            continue
        df = frames[key]
        if df.empty:
            issues.append(f"Required file is empty: {filename}")
            continue
        required = REQUIRED_COLUMNS.get(key, {"public_id"})
        missing_cols = sorted(required - set(df.columns))
        if missing_cols:
            issues.append(f"{filename} missing required columns: {missing_cols}")
        if "public_id" in df.columns:
            if df["public_id"].isna().any():
                issues.append(f"{filename} contains missing public_id values.")
            if key in {"demographics", "features"}:
                dup_n = int(df["public_id"].astype(str).duplicated().sum())
                if dup_n > 0:
                    issues.append(f"{filename} contains duplicated public_id values: {dup_n}")

    # Cross-file public_id consistency
    id_sets = {k: _id_set(v) for k, v in frames.items() if not v.empty and "public_id" in v.columns}
    if id_sets:
        common_ids = set.intersection(*id_sets.values())
        union_ids = set.union(*id_sets.values())
        if common_ids != union_ids:
            issues.append(
                "public_id sets differ across public files "
                f"(common={len(common_ids)}, union={len(union_ids)})."
            )
        for a, set_a in id_sets.items():
            for b, set_b in id_sets.items():
                if a >= b:
                    continue
                if set_a != set_b:
                    issues.append(
                        f"public_id mismatch: {PUBLIC_FILE_MAP[a]} vs {PUBLIC_FILE_MAP[b]} "
                        f"(a_only={len(set_a - set_b)}, b_only={len(set_b - set_a)})."
                    )
    else:
        common_ids = set()
        union_ids = set()

    # Demographics content checks
    demo = frames.get("demographics", pd.DataFrame())
    if not demo.empty and {"gender", "age", "public_id"}.issubset(demo.columns):
        gender_norm = normalize_gender_series(demo["gender"])
        invalid_gender = int(gender_norm.isna().sum())
        if invalid_gender > 0:
            issues.append(
                f"demographics_public.csv has non-normalizable gender values: {invalid_gender} rows."
            )
        age_num = pd.to_numeric(demo["age"], errors="coerce")
        invalid_age = int(age_num.isna().sum())
        if invalid_age > 0:
            issues.append(f"demographics_public.csv has non-numeric age values: {invalid_age} rows.")

    result = PublicValidationResult(
        ok=len(issues) == 0,
        n_common_ids=len(common_ids),
        n_union_ids=len(union_ids),
        issues=issues,
    )

    if raise_on_error and not result.ok:
        msg = "\n".join([f"- {issue}" for issue in result.issues])
        raise ValueError(f"Public data validation failed:\n{msg}")
    return result


def get_common_public_ids(validate: bool = True) -> set[str]:
    if validate:
        validate_public_bundle(raise_on_error=True)
    frames = load_public_frames()
    id_sets = [set(df["public_id"].dropna().astype(str)) for df in frames.values() if "public_id" in df.columns]
    if not id_sets:
        return set()
    return set.intersection(*id_sets)

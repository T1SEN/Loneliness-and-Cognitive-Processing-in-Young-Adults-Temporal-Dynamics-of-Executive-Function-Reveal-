"""Create the public-only de-identified data bundle from restricted inputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIR = Path("data") / "restricted_input"
DEFAULT_OUTPUT_DIR = Path("data") / "public"

SOURCE_FILES = {
    "surveys_public.csv": "2_surveys_results.csv",
    "features_public.csv": "5_overall_features.csv",
    "stroop_trials_public.csv": "4a_stroop_trials.csv",
    "wcst_trials_public.csv": "4b_wcst_trials.csv",
}
PARTICIPANTS_SOURCE = "1_participants_info.csv"

ID_COLUMNS = {"participantid", "participant_id"}
REMOVE_TOKENS = {"timestamp"}


def _normalize_col(name: str) -> str:
    return str(name).lstrip("\ufeff").strip()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_col(c) for c in df.columns]
    return df


def _find_id_column(df: pd.DataFrame, path: Path) -> str:
    for col in df.columns:
        if _normalize_col(col).lower() in ID_COLUMNS:
            return col
    raise ValueError(f"No participant ID column in {path}")


def _deterministic_public_id(participant_id: str, salt: int = 0) -> str:
    key = f"{participant_id}::{salt}" if salt else participant_id
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest().upper()
    return digest[:12]


def _build_id_map(ids: set[str]) -> dict[str, str]:
    id_map: dict[str, str] = {}
    used: set[str] = set()
    for pid in sorted(ids):
        salt = 0
        public_id = _deterministic_public_id(pid, salt=salt)
        while public_id in used:
            salt += 1
            public_id = _deterministic_public_id(pid, salt=salt)
        id_map[pid] = public_id
        used.add(public_id)
    return id_map


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return _normalize_columns(pd.read_csv(path, encoding="utf-8-sig"))


def _normalize_gender(value: object) -> str:
    token = "".join(ch for ch in str(value).strip().lower() if ch.isalpha() or ("\uac00" <= ch <= "\ud7a3"))
    male_exact = {"m", "male", "man", "men", "boy", "boys", "남", "남성", "남자"}
    female_exact = {"f", "female", "woman", "women", "girl", "girls", "여", "여성", "여자"}

    if token in male_exact:
        return "male"
    if token in female_exact:
        return "female"
    if "남성" in token or "남자" in token:
        return "male"
    if "여성" in token or "여자" in token:
        return "female"
    return ""


def _publicize_frame(df: pd.DataFrame, source_path: Path, id_map: dict[str, str]) -> pd.DataFrame:
    id_col = _find_id_column(df, source_path)

    out = df.copy()
    out[id_col] = out[id_col].astype(str)
    out = out[out[id_col].isin(id_map)].copy()

    keep_cols = []
    for col in out.columns:
        col_norm = _normalize_col(col).lower()
        if col_norm in ID_COLUMNS:
            continue
        if any(tok in col_norm for tok in REMOVE_TOKENS):
            continue
        keep_cols.append(col)

    out.insert(0, "public_id", out[id_col].map(id_map))
    out = out[["public_id"] + keep_cols]
    return out


def _build_demographics(participants_df: pd.DataFrame, source_path: Path, id_map: dict[str, str]) -> pd.DataFrame:
    id_col = _find_id_column(participants_df, source_path)

    gender_col = None
    age_col = None
    for col in participants_df.columns:
        low = _normalize_col(col).lower()
        if low == "gender":
            gender_col = col
        elif low == "age":
            age_col = col

    if gender_col is None or age_col is None:
        raise ValueError(f"Missing age/gender columns in {source_path}")

    demo = participants_df[[id_col, gender_col, age_col]].copy()
    demo[id_col] = demo[id_col].astype(str)
    demo = demo[demo[id_col].isin(id_map)].copy()
    demo = demo.drop_duplicates(subset=[id_col], keep="first")

    demo.insert(0, "public_id", demo[id_col].map(id_map))
    demo["gender"] = demo[gender_col].map(_normalize_gender)
    demo["age"] = pd.to_numeric(demo[age_col], errors="coerce")

    bad_gender = int((demo["gender"] == "").sum())
    if bad_gender > 0:
        raise ValueError(f"Could not normalize gender for {bad_gender} rows.")
    if demo["age"].isna().any():
        bad_age = int(demo["age"].isna().sum())
        raise ValueError(f"Non-numeric age values detected: {bad_age} rows.")

    demo = demo[["public_id", "gender", "age"]].copy()
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic public data bundle.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = [input_dir / src for src in SOURCE_FILES.values()]
    participants_path = input_dir / PARTICIPANTS_SOURCE
    source_paths.append(participants_path)

    id_sets: dict[Path, set[str]] = {}
    frames: dict[Path, pd.DataFrame] = {}

    for path in source_paths:
        df = _load_csv(path)
        id_col = _find_id_column(df, path)
        ids = set(df[id_col].dropna().astype(str))
        id_sets[path] = ids
        frames[path] = df

    if not id_sets:
        raise ValueError("No source files were loaded.")
    common_ids = set.intersection(*id_sets.values())
    if not common_ids:
        raise ValueError(
            "No common participant IDs across required source files. "
            "Cannot build a contract-consistent public bundle."
        )

    id_map = _build_id_map(common_ids)
    print(f"Common participant IDs across all source files: {len(common_ids)}")
    for path in source_paths:
        src_ids = id_sets[path]
        dropped = len(src_ids - common_ids)
        if dropped > 0:
            print(f"  - {path.name}: dropping {dropped} non-common IDs")

    for out_name, src_name in SOURCE_FILES.items():
        src_path = input_dir / src_name
        out_path = output_dir / out_name
        pub_df = _publicize_frame(frames[src_path], src_path, id_map)
        pub_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Wrote: {out_path} ({len(pub_df)} rows)")

    demo_df = _build_demographics(frames[participants_path], participants_path, id_map)
    demo_path = output_dir / "demographics_public.csv"
    demo_df.to_csv(demo_path, index=False, encoding="utf-8-sig")
    print(f"Wrote: {demo_path} ({len(demo_df)} rows)")


if __name__ == "__main__":
    main()

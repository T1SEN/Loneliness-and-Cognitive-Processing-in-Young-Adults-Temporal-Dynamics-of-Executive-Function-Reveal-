from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.analysis.utils import run_ucla_regression
from publication.preprocessing.constants import get_results_dir
from publication.preprocessing.core import ensure_participant_id
from publication.preprocessing.surveys import load_dass_scores, load_participants, load_ucla_scores
from publication.preprocessing.wcst._shared import prepare_wcst_trials


OUTPUT_DIR = ROOT / "publication" / "data" / "outputs" / "analysis" / "reviewer_response"
N_CATEGORIES = 6
COV_TYPE = "nonrobust"


def add_zscores(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        std = df[col].std()
        df[f"z_{col}"] = (df[col] - df[col].mean()) / std if std and np.isfinite(std) else np.nan
    return df


def load_base_data() -> pd.DataFrame:
    data_dir = get_results_dir("overall")
    participants = load_participants(data_dir)
    ucla = load_ucla_scores(data_dir).rename(columns={"ucla_total": "ucla_score"})
    dass = load_dass_scores(data_dir)

    base = participants.merge(ucla, on="participant_id", how="inner")
    base = base.merge(dass, on="participant_id", how="left")
    base["gender_male"] = base["gender"].map({"male": 1, "female": 0})
    return base


def load_qc_ids(task: str) -> set[str]:
    task_dir = get_results_dir(task)
    qc_ids_path = task_dir / "filtered_participant_ids.csv"
    if not qc_ids_path.exists():
        return set()
    qc_ids = pd.read_csv(qc_ids_path, encoding="utf-8-sig")
    qc_ids = ensure_participant_id(qc_ids)
    if "participant_id" not in qc_ids.columns:
        return set()
    return set(qc_ids["participant_id"].dropna().astype(str))


def assign_category_numbers(
    wcst: pd.DataFrame,
    rule_col: str,
    trial_col: str,
) -> pd.DataFrame:
    df = wcst.sort_values(["participant_id", trial_col]).copy()
    df["category_num"] = np.nan

    for pid, grp in df.groupby("participant_id"):
        grp_sorted = grp.sort_values(trial_col).copy()
        idxs = grp_sorted.index.to_list()
        rules = (
            grp_sorted[rule_col].astype(str).str.lower().replace({"color": "colour"}).to_numpy()
        )

        change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
        segment_starts = [0] + change_indices
        segment_ends = change_indices + [len(rules)]

        for cat_idx, (start, end) in enumerate(zip(segment_starts, segment_ends), start=1):
            if cat_idx > N_CATEGORIES:
                break
            if start >= end:
                continue
            for i in range(start, end):
                df.at[idxs[i], "category_num"] = float(cat_idx)

    return df


def compute_category_slope(df: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for pid, grp in df.groupby("participant_id"):
        means = (
            grp.groupby("category_num")["rt_ms"].mean().dropna().sort_index()
        )
        if len(means) < min_n:
            slope = np.nan
        else:
            x = means.index.to_numpy(dtype=float)
            y = means.to_numpy(dtype=float)
            slope = float(np.polyfit(x, y, 1)[0])
        records.append({
            "participant_id": pid,
            "wcst_category_rt_slope": slope,
            "wcst_category_rt_n": float(len(means)),
        })
    return pd.DataFrame(records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = load_base_data()
    base = add_zscores(
        base,
        ["ucla_score", "dass_depression", "dass_anxiety", "dass_stress", "age"],
    )

    prepared = prepare_wcst_trials()
    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]
    trial_col = prepared["trial_col"]
    rule_col = prepared["rule_col"]
    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None or trial_col is None or rule_col is None:
        raise RuntimeError("WCST trials missing required columns.")

    qc_ids = load_qc_ids("wcst")
    if qc_ids:
        wcst = wcst[wcst["participant_id"].isin(qc_ids)]

    wcst["rt_ms"] = pd.to_numeric(wcst[rt_col], errors="coerce")
    wcst = wcst[wcst["rt_ms"].notna()]
    if "is_rt_valid" in wcst.columns:
        wcst = wcst[wcst["is_rt_valid"] == True]

    wcst["rule"] = wcst[rule_col].astype(str).str.lower().replace({"color": "colour"})
    wcst = assign_category_numbers(wcst, rule_col="rule", trial_col=trial_col)
    wcst = wcst.dropna(subset=["category_num", "rt_ms"])

    overall_mean = (
        wcst.groupby("participant_id")["rt_ms"].mean().reset_index().rename(
            columns={"rt_ms": "wcst_overall_rt_mean"}
        )
    )
    category_slope = compute_category_slope(wcst)

    summary = overall_mean.merge(category_slope, on="participant_id", how="outer")
    summary = summary.merge(base, on="participant_id", how="inner")

    results: list[dict[str, object]] = []

    res_mean = run_ucla_regression(summary, "wcst_overall_rt_mean", cov_type=COV_TYPE)
    if res_mean:
        res_mean["outcome"] = "WCST Overall RT Mean"
        res_mean["outcome_type"] = "overall_mean"
        results.append(res_mean)

    res_slope = run_ucla_regression(summary, "wcst_category_rt_slope", cov_type=COV_TYPE)
    if res_slope:
        res_slope["outcome"] = "WCST Category RT Slope (mean RT vs category)"
        res_slope["outcome_type"] = "category_slope"
        results.append(res_slope)

    results_df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "wcst_overall_category_rt_ols.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(results_df[["outcome", "n", "ucla_beta", "ucla_p", "cov_type"]].to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

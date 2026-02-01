from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from static.analysis.utils import get_output_dir, run_ucla_regression
from static.preprocessing.constants import get_results_dir
from static.preprocessing.core import ensure_participant_id
from static.preprocessing.surveys import load_dass_scores, load_participants, load_ucla_scores
from static.preprocessing.wcst.phase import label_wcst_phases
from static.preprocessing.wcst.utils import prepare_wcst_trials


PHASE_MAP = {
    "exploration": "rule_search",
    "confirmation": "rule_application",
    "exploitation": "rule_application",
}


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


def prepare_rule_phase_means(
    wcst: pd.DataFrame,
    rt_col: str,
    trial_col: str,
    rule_col: str,
    confirm_len: int,
    min_trials: int,
    include_errors: bool,
) -> pd.DataFrame:
    wcst = wcst.copy()
    wcst["rt_ms"] = pd.to_numeric(wcst[rt_col], errors="coerce")
    wcst = wcst[wcst["rt_ms"].notna()]
    if "is_rt_valid" in wcst.columns:
        wcst = wcst[wcst["is_rt_valid"] == True]

    wcst["rule"] = wcst[rule_col].astype(str).str.lower().replace({"color": "colour"})
    wcst = label_wcst_phases(wcst, rule_col="rule", trial_col=trial_col, confirm_len=confirm_len)
    wcst["rule_phase"] = wcst["phase"].map(PHASE_MAP)
    wcst = wcst.dropna(subset=["rule_phase", "category_num"])

    if not include_errors:
        wcst = wcst[wcst["correct"].astype(bool)]

    phase_counts = (
        wcst.groupby(["participant_id", "category_num", "rule_phase"], observed=False)["rt_ms"]
        .size()
        .reset_index(name="n_trials")
    )
    phase_counts_wide = phase_counts.pivot(
        index=["participant_id", "category_num"],
        columns="rule_phase",
        values="n_trials",
    )
    phase_counts_wide = phase_counts_wide.fillna(0)
    valid_categories = phase_counts_wide[
        (phase_counts_wide.get("rule_search", 0) >= min_trials)
        & (phase_counts_wide.get("rule_application", 0) >= min_trials)
    ]
    if valid_categories.empty:
        return pd.DataFrame()

    valid_keys = valid_categories.reset_index()[["participant_id", "category_num"]]
    wcst = wcst.merge(valid_keys, on=["participant_id", "category_num"], how="inner")

    phase_means = (
        wcst.groupby(["participant_id", "rule_phase"], observed=False)["rt_ms"]
        .mean()
        .unstack()
        .reset_index()
    )
    if "rule_search" not in phase_means.columns:
        phase_means["rule_search"] = np.nan
    if "rule_application" not in phase_means.columns:
        phase_means["rule_application"] = np.nan

    phase_means = phase_means.rename(
        columns={
            "rule_search": "rule_search_rt_mean",
            "rule_application": "rule_application_rt_mean",
        }
    )
    phase_means["rule_search_minus_application"] = (
        phase_means["rule_search_rt_mean"] - phase_means["rule_application_rt_mean"]
    )
    return phase_means


def main(confirm_len: int, min_trials: int, include_errors: bool) -> None:
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

    qc_ids = load_qc_ids("overall")
    if qc_ids:
        wcst = wcst[wcst["participant_id"].isin(qc_ids)]

    phase_means = prepare_rule_phase_means(
        wcst,
        rt_col=rt_col,
        trial_col=trial_col,
        rule_col=rule_col,
        confirm_len=confirm_len,
        min_trials=min_trials,
        include_errors=include_errors,
    )
    if phase_means.empty:
        raise RuntimeError("No valid phase means after filtering.")

    predictors = base[
        [
            "participant_id",
            "z_ucla_score",
            "z_dass_depression",
            "z_dass_anxiety",
            "z_dass_stress",
            "z_age",
            "gender_male",
        ]
    ].dropna()
    data = phase_means.merge(predictors, on="participant_id", how="inner")

    outcomes = [
        ("rule_search_rt_mean", "rule_search_rt_mean", "phase_mean"),
        ("rule_application_rt_mean", "rule_application_rt_mean", "phase_mean"),
        ("rule_search_minus_application", "rule_search_minus_application", "phase_contrast"),
    ]

    results: list[dict[str, object]] = []
    for label, col, out_type in outcomes:
        res = run_ucla_regression(data, col, cov_type="nonrobust")
        if res is None:
            continue
        res["outcome"] = label
        res["outcome_type"] = out_type
        res["confirm_len"] = confirm_len
        res["min_trials"] = min_trials
        res["include_errors"] = include_errors
        results.append(res)

    results_df = pd.DataFrame(results)
    output_dir = get_output_dir("overall")
    suffix = "_alltrials" if include_errors else ""
    if confirm_len == 3:
        filename = f"wcst_rule_search_application_rt_ols{suffix}.csv"
    else:
        filename = f"wcst_rule_search_application_rt_ols_m{confirm_len}{suffix}.csv"
    out_path = output_dir / filename
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    if not results_df.empty:
        print(results_df[["outcome", "n", "ucla_beta", "ucla_p", "cov_type"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-len", type=int, default=3)
    parser.add_argument("--min-trials", type=int, default=2)
    parser.add_argument("--include-errors", action="store_true")
    args = parser.parse_args()
    main(args.confirm_len, args.min_trials, args.include_errors)


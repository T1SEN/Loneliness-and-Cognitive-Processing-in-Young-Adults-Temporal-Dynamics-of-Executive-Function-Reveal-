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
from static.wcst_phase_utils import prepare_wcst_trials

from static.wcst_phase_utils import label_wcst_phases


PHASE_ORDER = ["exploration", "confirmation", "exploitation"]


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


def _prepare_phase_means(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    phase_means = (
        df.groupby(["participant_id", "phase"], observed=False)[value_col]
        .mean()
        .unstack()
        .reset_index()
    )
    for phase in PHASE_ORDER:
        if phase not in phase_means.columns:
            phase_means[phase] = np.nan
    phase_means = phase_means.rename(
        columns={
            "exploration": f"{prefix}_exploration",
            "confirmation": f"{prefix}_confirmation",
            "exploitation": f"{prefix}_exploitation",
        }
    )
    phase_means["exploration_minus_exploitation"] = (
        phase_means[f"{prefix}_exploration"] - phase_means[f"{prefix}_exploitation"]
    )
    phase_means["confirmation_minus_exploitation"] = (
        phase_means[f"{prefix}_confirmation"] - phase_means[f"{prefix}_exploitation"]
    )
    phase_means["exploration_minus_confirmation"] = (
        phase_means[f"{prefix}_exploration"] - phase_means[f"{prefix}_confirmation"]
    )
    return phase_means


def _prepare_pre_exploit_means(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df["merged_phase"] = df["phase"].map(
        {
            "exploration": "pre_exploitation",
            "confirmation": "pre_exploitation",
            "exploitation": "exploitation",
        }
    )
    phase_means = (
        df.groupby(["participant_id", "merged_phase"], observed=False)[value_col]
        .mean()
        .unstack()
        .reset_index()
    )
    if "pre_exploitation" not in phase_means.columns:
        phase_means["pre_exploitation"] = np.nan
    if "exploitation" not in phase_means.columns:
        phase_means["exploitation"] = np.nan
    phase_means = phase_means.rename(
        columns={
            "pre_exploitation": f"{prefix}_pre_exploitation",
            "exploitation": f"{prefix}_exploitation",
        }
    )
    phase_means["pre_exploitation_minus_exploitation"] = (
        phase_means[f"{prefix}_pre_exploitation"] - phase_means[f"{prefix}_exploitation"]
    )
    return phase_means


def main(confirm_len: int, include_errors: bool, use_log: bool, merge_pre: bool) -> None:
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

    wcst["rt_ms"] = pd.to_numeric(wcst[rt_col], errors="coerce")
    wcst = wcst[wcst["rt_ms"].notna()]
    if "is_rt_valid" in wcst.columns:
        wcst = wcst[wcst["is_rt_valid"] == True]

    wcst["rule"] = wcst[rule_col].astype(str).str.lower().replace({"color": "colour"})
    wcst = label_wcst_phases(wcst, rule_col="rule", trial_col=trial_col, confirm_len=confirm_len)
    if not include_errors:
        wcst = wcst[wcst["correct"].astype(bool)]

    if use_log:
        wcst["log_rt"] = np.where(wcst["rt_ms"] > 0, np.log(wcst["rt_ms"]), np.nan)
        value_col = "log_rt"
        prefix = "log_rt"
    else:
        value_col = "rt_ms"
        prefix = "rt_ms"
    wcst = wcst.dropna(subset=[value_col, "phase"])

    if merge_pre:
        phase_means = _prepare_pre_exploit_means(wcst, value_col=value_col, prefix=prefix)
    else:
        phase_means = _prepare_phase_means(wcst, value_col=value_col, prefix=prefix)
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

    if merge_pre:
        outcomes = [
            ("pre_exploitation", f"{prefix}_pre_exploitation", "phase_mean"),
            ("exploitation", f"{prefix}_exploitation", "phase_mean"),
            ("pre_exploitation_minus_exploitation", "pre_exploitation_minus_exploitation", "phase_contrast"),
        ]
    else:
        outcomes = [
            ("exploration", f"{prefix}_exploration", "phase_mean"),
            ("confirmation", f"{prefix}_confirmation", "phase_mean"),
            ("exploitation", f"{prefix}_exploitation", "phase_mean"),
            ("exploration_minus_exploitation", "exploration_minus_exploitation", "phase_contrast"),
            ("confirmation_minus_exploitation", "confirmation_minus_exploitation", "phase_contrast"),
            ("exploration_minus_confirmation", "exploration_minus_confirmation", "phase_contrast"),
        ]

    results: list[dict[str, object]] = []
    for label, col, out_type in outcomes:
        res = run_ucla_regression(data, col, cov_type="nonrobust")
        if res is None:
            continue
        res["outcome"] = label
        res["outcome_type"] = out_type
        res["confirm_len"] = confirm_len
        res["include_errors"] = include_errors
        res["use_log"] = use_log
        res["merge_pre"] = merge_pre
        results.append(res)

    results_df = pd.DataFrame(results)
    output_dir = get_output_dir("overall")
    if use_log and not include_errors and not merge_pre:
        if confirm_len == 3:
            filename = "wcst_phase_rt_log_ols.csv"
        else:
            filename = f"wcst_phase_rt_log_ols_m{confirm_len}.csv"
    else:
        if merge_pre:
            base_name = "wcst_phase_pre_exploit_rt_log_ols" if use_log else "wcst_phase_pre_exploit_rt_ols"
        else:
            base_name = "wcst_phase_rt_log_ols" if use_log else "wcst_phase_rt_ols"
        if confirm_len == 3:
            filename = base_name
        else:
            filename = f"{base_name}_m{confirm_len}"
        if include_errors:
            filename = f"{filename}_alltrials"
        filename = f"{filename}.csv"
    out_path = output_dir / filename
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    if not results_df.empty:
        print(results_df[["outcome", "n", "ucla_beta", "ucla_p", "cov_type"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-len", type=int, default=3)
    parser.add_argument("--include-errors", action="store_true")
    parser.add_argument("--use-log", action="store_true")
    parser.add_argument("--merge-pre", action="store_true")
    args = parser.parse_args()
    main(args.confirm_len, args.include_errors, args.use_log, args.merge_pre)



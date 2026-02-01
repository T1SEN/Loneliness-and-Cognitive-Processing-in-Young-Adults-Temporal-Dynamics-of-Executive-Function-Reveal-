from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.analysis.utils import get_output_dir, run_ucla_regression
from publication.preprocessing.constants import get_results_dir
from publication.preprocessing.core import ensure_participant_id
from publication.preprocessing.surveys import load_dass_scores, load_participants, load_ucla_scores
from publication.wcst_phase_utils import prepare_wcst_trials

from run_wcst_post_shift_error_log_ols import label_wcst_phases


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


def _summarize_counts(values: pd.Series) -> dict[str, float]:
    if values.empty:
        return {
            "n_participants": 0,
            "total_trials": 0,
            "mean_trials": np.nan,
            "median_trials": np.nan,
            "sd_trials": np.nan,
            "min_trials": np.nan,
            "max_trials": np.nan,
            "q25_trials": np.nan,
            "q75_trials": np.nan,
        }
    return {
        "n_participants": int(values.count()),
        "total_trials": float(values.sum()),
        "mean_trials": float(values.mean()),
        "median_trials": float(values.median()),
        "sd_trials": float(values.std(ddof=1)) if values.count() > 1 else np.nan,
        "min_trials": float(values.min()),
        "max_trials": float(values.max()),
        "q25_trials": float(values.quantile(0.25)),
        "q75_trials": float(values.quantile(0.75)),
    }


def main(confirm_len: int) -> None:
    base = load_base_data()
    base = add_zscores(
        base,
        ["ucla_score", "dass_depression", "dass_anxiety", "dass_stress", "age"],
    )
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
    wcst["log_rt"] = np.where(wcst["rt_ms"] > 0, np.log(wcst["rt_ms"]), np.nan)
    wcst = wcst.dropna(subset=["log_rt", "phase", "category_num"])

    # Participant x category x phase summaries
    phase_cat = (
        wcst.groupby(["participant_id", "category_num", "phase"], observed=False)
        .agg(
            mean_log_rt=("log_rt", "mean"),
            n_trials=("log_rt", "size"),
        )
        .reset_index()
    )

    output_dir = get_output_dir("overall")
    if confirm_len == 3:
        prefix = "wcst_phase_category"
    else:
        prefix = f"wcst_phase_category_m{confirm_len}"

    phase_cat_path = output_dir / f"{prefix}_rt_summary.csv"
    phase_cat.to_csv(phase_cat_path, index=False, encoding="utf-8-sig")

    # Phase trial count distributions (per participant)
    phase_counts = (
        phase_cat.groupby(["participant_id", "phase"], observed=False)["n_trials"]
        .sum()
        .reset_index()
    )
    phase_counts_wide = phase_counts.pivot(index="participant_id", columns="phase", values="n_trials")

    count_rows = []
    for phase in PHASE_ORDER:
        if phase not in phase_counts_wide.columns:
            continue
        vals = phase_counts_wide[phase].dropna()
        stats = _summarize_counts(vals)
        count_rows.append({"phase": phase, **stats})

    phase_count_summary = pd.DataFrame(count_rows)
    phase_count_path = output_dir / f"{prefix}_trial_count_summary.csv"
    phase_count_summary.to_csv(phase_count_path, index=False, encoding="utf-8-sig")

    # Phase x category trial count summaries
    count_by_cat = (
        phase_cat.groupby(["phase", "category_num"], observed=False)["n_trials"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_trials",
                "median": "median_trials",
                "min": "min_trials",
                "max": "max_trials",
                "count": "n_participants",
            }
        )
    )
    count_by_cat_path = output_dir / f"{prefix}_trial_count_by_category.csv"
    count_by_cat.to_csv(count_by_cat_path, index=False, encoding="utf-8-sig")

    # Category slope (per participant, per phase)
    slope_records = []
    for (pid, phase), grp in phase_cat.groupby(["participant_id", "phase"], observed=False):
        vals = grp.dropna(subset=["mean_log_rt"])
        if len(vals) < 3:
            continue
        x = vals["category_num"].to_numpy(dtype=float)
        y = vals["mean_log_rt"].to_numpy(dtype=float)
        if np.isfinite(y).sum() < 3:
            continue
        slope = float(np.polyfit(x, y, 1)[0])
        slope_records.append({
            "participant_id": pid,
            "phase": phase,
            "category_slope": slope,
            "n_categories": int(len(vals)),
        })

    slope_df = pd.DataFrame(slope_records)
    slope_path = output_dir / f"{prefix}_slope_by_participant.csv"
    slope_df.to_csv(slope_path, index=False, encoding="utf-8-sig")

    # OLS on category slopes
    slope_wide = slope_df.pivot(index="participant_id", columns="phase", values="category_slope").reset_index()
    slope_wide = slope_wide.rename(
        columns={
            "exploration": "exploration_slope",
            "confirmation": "confirmation_slope",
            "exploitation": "exploitation_slope",
        }
    )
    slope_data = slope_wide.merge(predictors, on="participant_id", how="inner")

    slope_results = []
    for phase, col in [
        ("exploration", "exploration_slope"),
        ("confirmation", "confirmation_slope"),
        ("exploitation", "exploitation_slope"),
    ]:
        if col not in slope_data.columns:
            continue
        res = run_ucla_regression(slope_data, col, cov_type="nonrobust")
        if res is None:
            continue
        res["phase"] = phase
        res["outcome"] = "category_slope"
        slope_results.append(res)

    slope_results_df = pd.DataFrame(slope_results)
    slope_results_path = output_dir / f"{prefix}_slope_ols.csv"
    slope_results_df.to_csv(slope_results_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {phase_cat_path}")
    print(f"Saved: {phase_count_path}")
    print(f"Saved: {count_by_cat_path}")
    print(f"Saved: {slope_path}")
    print(f"Saved: {slope_results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-len", type=int, default=3)
    args = parser.parse_args()
    main(args.confirm_len)

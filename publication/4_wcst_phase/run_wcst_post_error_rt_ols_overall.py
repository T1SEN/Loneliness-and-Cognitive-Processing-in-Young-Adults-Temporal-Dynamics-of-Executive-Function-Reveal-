from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.preprocessing.constants import WCST_RT_MAX, WCST_RT_MIN, get_results_dir
from publication.preprocessing.core import ensure_participant_id
from publication.preprocessing.surveys import load_dass_scores, load_participants, load_ucla_scores
from publication.wcst_phase_utils import prepare_wcst_trials


OUTPUT_DIR = ROOT / "publication" / "data" / "outputs" / "analysis" / "overall" / "wcst_phase"


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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None or trial_col is None:
        raise RuntimeError("WCST trials missing required columns.")

    qc_ids = load_qc_ids("overall")
    if qc_ids:
        wcst = wcst[wcst["participant_id"].isin(qc_ids)]

    wcst["rt_ms_raw"] = pd.to_numeric(wcst[rt_col], errors="coerce")
    wcst["rt_ms"] = wcst["rt_ms_raw"]
    if "is_rt_valid" in wcst.columns:
        valid_mask = wcst["is_rt_valid"].astype(bool)
    else:
        valid_mask = wcst["rt_ms"].between(WCST_RT_MIN, WCST_RT_MAX)
    wcst["rt_ms"] = wcst["rt_ms"].where(valid_mask, np.nan)

    wcst["correct"] = wcst["correct"].astype(bool)
    wcst = wcst.sort_values(["participant_id", trial_col]).copy()
    wcst["prev_correct"] = wcst.groupby("participant_id")["correct"].shift(1)
    wcst["post_error"] = (wcst["prev_correct"] == False).fillna(False)

    post_error_trials = wcst[wcst["post_error"]].copy()
    post_error_means = (
        post_error_trials.groupby("participant_id")["rt_ms"].mean().rename("post_error_rt_mean")
    )
    overall_means_all = wcst.groupby("participant_id")["rt_ms_raw"].mean().rename("overall_rt_mean_all")

    shift_records = []
    for pid, grp in wcst.groupby("participant_id"):
        if trial_col and trial_col in grp.columns:
            grp = grp.sort_values(trial_col)
        rules = grp[prepared["rule_col"]].astype(str).str.lower().replace({"color": "colour"}).values
        rt_vals = pd.to_numeric(grp["rt_ms"], errors="coerce").astype(float).values
        correct = grp["correct"].astype(bool).values

        change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
        n_switches = len(change_indices)
        shift_post_error_rts = []
        for s_idx, idx in enumerate(change_indices):
            next_idx = change_indices[s_idx + 1] if s_idx + 1 < n_switches else len(rt_vals)
            reacq_idx = None
            for j in range(idx, next_idx - 2):
                if correct[j] and correct[j + 1] and correct[j + 2]:
                    reacq_idx = j + 2
                    break
            shift_end = reacq_idx if reacq_idx is not None else next_idx - 1
            if shift_end >= idx:
                for j in range(idx, shift_end + 1):
                    if correct[j]:
                        continue
                    if j + 1 < next_idx:
                        rt_next = rt_vals[j + 1]
                        if np.isfinite(rt_next):
                            shift_post_error_rts.append(float(rt_next))
        mean_val = float(np.mean(shift_post_error_rts)) if shift_post_error_rts else np.nan
        shift_records.append({
            "participant_id": pid,
            "post_shift_error_rt_mean": mean_val,
        })
    post_shift_means = pd.DataFrame(shift_records)

    summary = pd.DataFrame({"participant_id": wcst["participant_id"].unique()})
    summary = summary.merge(post_error_means, on="participant_id", how="left")
    summary = summary.merge(post_shift_means, on="participant_id", how="left")
    summary = summary.merge(overall_means_all, on="participant_id", how="left")

    data = summary.merge(predictors, on="participant_id", how="inner")
    data_post_error = data.dropna(subset=["post_error_rt_mean", "overall_rt_mean_all"]).copy()
    if data_post_error.empty:
        raise RuntimeError("No participants with post-error RT and overall RT.")

    formula = (
        "post_error_rt_mean ~ z_ucla_score + overall_rt_mean_all + "
        "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )
    model = smf.ols(formula, data=data_post_error).fit()

    out = {
        "outcome": "post_error_rt_mean",
        "rt_min_ms": int(WCST_RT_MIN),
        "rt_max_ms": int(WCST_RT_MAX),
        "n_participants": int(data_post_error["participant_id"].nunique()),
        "n_post_error_trials": int(len(post_error_trials)),
        "ucla_beta": float(model.params.get("z_ucla_score", np.nan)),
        "ucla_se": float(model.bse.get("z_ucla_score", np.nan)),
        "ucla_t": float(model.tvalues.get("z_ucla_score", np.nan)),
        "ucla_p": float(model.pvalues.get("z_ucla_score", np.nan)),
        "overall_rt_all_beta": float(model.params.get("overall_rt_mean_all", np.nan)),
        "overall_rt_all_se": float(model.bse.get("overall_rt_mean_all", np.nan)),
        "overall_rt_all_t": float(model.tvalues.get("overall_rt_mean_all", np.nan)),
        "overall_rt_all_p": float(model.pvalues.get("overall_rt_mean_all", np.nan)),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
    }

    data_post_shift = data.dropna(subset=["post_shift_error_rt_mean", "overall_rt_mean_all"]).copy()
    if data_post_shift.empty:
        raise RuntimeError("No participants with post-shift error RT and overall RT.")

    formula = (
        "post_shift_error_rt_mean ~ z_ucla_score + overall_rt_mean_all + "
        "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )
    model = smf.ols(formula, data=data_post_shift).fit()

    out_shift = {
        "outcome": "post_shift_error_rt_mean",
        "rt_min_ms": int(WCST_RT_MIN),
        "rt_max_ms": int(WCST_RT_MAX),
        "n_participants": int(data_post_shift["participant_id"].nunique()),
        "n_post_error_trials": int(len(post_error_trials)),
        "ucla_beta": float(model.params.get("z_ucla_score", np.nan)),
        "ucla_se": float(model.bse.get("z_ucla_score", np.nan)),
        "ucla_t": float(model.tvalues.get("z_ucla_score", np.nan)),
        "ucla_p": float(model.pvalues.get("z_ucla_score", np.nan)),
        "overall_rt_all_beta": float(model.params.get("overall_rt_mean_all", np.nan)),
        "overall_rt_all_se": float(model.bse.get("overall_rt_mean_all", np.nan)),
        "overall_rt_all_t": float(model.tvalues.get("overall_rt_mean_all", np.nan)),
        "overall_rt_all_p": float(model.pvalues.get("overall_rt_mean_all", np.nan)),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
    }

    out_df = pd.DataFrame([out, out_shift])
    out_path = OUTPUT_DIR / "wcst_post_error_rt_ols_overall_rt_all_trials.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out_df.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()



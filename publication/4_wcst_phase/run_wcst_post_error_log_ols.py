from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.preprocessing.constants import get_results_dir
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

    prepared = prepare_wcst_trials()
    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]
    trial_col = prepared["trial_col"]
    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None or trial_col is None:
        raise RuntimeError("WCST trials missing required columns.")

    qc_ids = load_qc_ids("overall")
    if qc_ids:
        wcst = wcst[wcst["participant_id"].isin(qc_ids)]

    wcst["rt_ms"] = pd.to_numeric(wcst[rt_col], errors="coerce")
    wcst = wcst[wcst["rt_ms"].notna()]
    if "is_rt_valid" in wcst.columns:
        wcst = wcst[wcst["is_rt_valid"] == True]

    wcst = wcst.sort_values(["participant_id", trial_col]).copy()
    wcst["prev_correct"] = wcst.groupby("participant_id")["correct"].shift(1)
    wcst["post_error"] = wcst["prev_correct"] == False
    wcst["post_error"] = wcst["post_error"].astype("boolean").fillna(False)
    wcst["log_rt"] = np.where(wcst["rt_ms"] > 0, np.log(wcst["rt_ms"]), np.nan)
    wcst = wcst.dropna(subset=["log_rt"])

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

    trials = wcst.merge(predictors, on="participant_id", how="inner")
    trials = trials[trials["post_error"]].copy()

    means = (
        trials.groupby("participant_id")["log_rt"].mean().reset_index().rename(
            columns={"log_rt": "post_error_log_rt_mean"}
        )
    )
    data = means.merge(predictors, on="participant_id", how="inner")

    formula = (
        "post_error_log_rt_mean ~ z_ucla_score + "
        "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )
    model = smf.ols(formula, data=data).fit()

    out = {
        "n_participants": int(data["participant_id"].nunique()),
        "n_trials": int(len(trials)),
        "ucla_beta": float(model.params.get("z_ucla_score", np.nan)),
        "ucla_se": float(model.bse.get("z_ucla_score", np.nan)),
        "ucla_t": float(model.tvalues.get("z_ucla_score", np.nan)),
        "ucla_p": float(model.pvalues.get("z_ucla_score", np.nan)),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
    }

    out_df = pd.DataFrame([out])
    out_path = OUTPUT_DIR / "wcst_post_error_log_ols.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out_df.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()



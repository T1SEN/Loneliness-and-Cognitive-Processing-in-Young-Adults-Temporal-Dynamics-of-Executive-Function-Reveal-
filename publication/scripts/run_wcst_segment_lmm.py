from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.preprocessing.constants import get_results_dir
from publication.preprocessing.core import ensure_participant_id
from publication.preprocessing.surveys import load_dass_scores, load_participants, load_ucla_scores
from publication.preprocessing.wcst._shared import prepare_wcst_trials


OUTPUT_DIR = ROOT / "publication" / "data" / "outputs" / "analysis" / "reviewer_response"
N_CATEGORIES = 6


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


def label_wcst_segments(
    wcst: pd.DataFrame,
    rule_col: str,
    trial_col: str,
) -> pd.DataFrame:
    df = wcst.sort_values(["participant_id", trial_col]).copy()
    df["category_num"] = np.nan
    df["phase"] = pd.NA
    df["trial_in_category"] = np.nan
    df["shift_trial"] = False
    df["post_shift_error"] = False

    for pid, grp in df.groupby("participant_id"):
        grp_sorted = grp.sort_values(trial_col).copy()
        idxs = grp_sorted.index.to_list()
        rules = (
            grp_sorted[rule_col].astype(str).str.lower().replace({"color": "colour"}).to_numpy()
        )
        correct = grp_sorted["correct"].astype(bool).to_numpy()

        change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
        segment_starts = [0] + change_indices
        segment_ends = change_indices + [len(rules)]

        for cat_idx, (start, end) in enumerate(zip(segment_starts, segment_ends), start=1):
            if cat_idx > N_CATEGORIES:
                break
            if start >= end:
                continue

            reacq_start = None
            reacq_idx = None
            for j in range(start, end - 2):
                if correct[j] and correct[j + 1] and correct[j + 2]:
                    reacq_start = j
                    reacq_idx = j + 2
                    break
            shift_end_err = reacq_idx if reacq_idx is not None else end - 1

            for i in range(start, end):
                row_idx = idxs[i]
                df.at[row_idx, "category_num"] = float(cat_idx)
                df.at[row_idx, "trial_in_category"] = float(i - start + 1)
                if i == start:
                    df.at[row_idx, "shift_trial"] = True

                if reacq_start is None:
                    df.at[row_idx, "phase"] = "exploration"
                elif i < reacq_start:
                    df.at[row_idx, "phase"] = "exploration"
                elif i <= reacq_idx:
                    df.at[row_idx, "phase"] = "confirmation"
                else:
                    df.at[row_idx, "phase"] = "exploitation"

            for j in range(start, shift_end_err + 1):
                if not correct[j] and j + 1 < end:
                    df.at[idxs[j + 1], "post_shift_error"] = True

    return df


def _fit_mixedlm(
    formula: str,
    df: pd.DataFrame,
    re_formula: str,
    method: str,
) -> tuple[object, list[str]]:
    warning_msgs: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = smf.mixedlm(
            formula,
            data=df,
            groups=df["participant_id"],
            re_formula=re_formula,
        )
        result = model.fit(reml=False, method=method, maxiter=200)
    for warn in caught:
        if issubclass(warn.category, ConvergenceWarning):
            warning_msgs.append(str(warn.message))
    return result, warning_msgs


def fit_with_fallback(
    formula: str,
    df: pd.DataFrame,
    re_formulas: list[str],
) -> tuple[object, str, str, list[str], str | None]:
    last_result = None
    last_re = re_formulas[-1]
    last_method = "lbfgs"
    last_warnings: list[str] = []
    last_note: str | None = None
    for re_formula in re_formulas:
        for method in ("lbfgs", "powell"):
            try:
                result, warning_msgs = _fit_mixedlm(formula, df, re_formula, method)
                last_result = result
                last_re = re_formula
                last_method = method
                last_warnings = warning_msgs
                last_note = None
                if getattr(result, "converged", False) and not warning_msgs:
                    return result, re_formula, method, warning_msgs, None
            except Exception as exc:
                last_note = str(exc)
                continue
    if last_result is None:
        raise RuntimeError("All MixedLM attempts failed.")
    return last_result, last_re, last_method, last_warnings, last_note


def _get_term(result: object, term: str) -> dict[str, float]:
    params = getattr(result, "params", {})
    bse = getattr(result, "bse", {})
    tvalues = getattr(result, "tvalues", {})
    pvalues = getattr(result, "pvalues", {})
    return {
        "beta": float(params.get(term, np.nan)),
        "se": float(bse.get(term, np.nan)),
        "z": float(tvalues.get(term, np.nan)),
        "p": float(pvalues.get(term, np.nan)),
    }


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
    wcst = label_wcst_segments(wcst, rule_col="rule", trial_col=trial_col)
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

    df = wcst.merge(predictors, on="participant_id", how="inner")
    if df.empty:
        raise RuntimeError("No WCST trials available after merging predictors.")

    formula = (
        "log_rt ~ z_ucla_score + z_dass_depression + "
        "z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )

    segment_specs = [
        ("shift_trial", df["shift_trial"] == True),
        ("post_shift_error", df["post_shift_error"] == True),
        ("exploration", df["phase"] == "exploration"),
        ("confirmation", df["phase"] == "confirmation"),
        ("exploitation", df["phase"] == "exploitation"),
    ]

    rows: list[dict[str, object]] = []
    for label, mask in segment_specs:
        subset = df[mask].copy()
        if subset.empty:
            rows.append({
                "segment": label,
                "n_trials": 0,
                "n_participants": 0,
                "re_formula": "1",
                "method": None,
                "converged": False,
                "ucla_beta": np.nan,
                "ucla_se": np.nan,
                "ucla_z": np.nan,
                "ucla_p": np.nan,
                "fit_note": "empty_subset",
            })
            continue

        result, re_formula, method, warn_msgs, fit_note = fit_with_fallback(
            formula,
            subset,
            ["1"],
        )
        ucla = _get_term(result, "z_ucla_score")
        rows.append({
            "segment": label,
            "n_trials": int(len(subset)),
            "n_participants": int(subset["participant_id"].nunique()),
            "re_formula": re_formula,
            "method": method,
            "converged": bool(getattr(result, "converged", False)),
            "ucla_beta": ucla["beta"],
            "ucla_se": ucla["se"],
            "ucla_z": ucla["z"],
            "ucla_p": ucla["p"],
            "warning_count": len(warn_msgs),
            "fit_note": fit_note,
        })

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "wcst_segment_lmm_ols_equivalent.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out_df.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

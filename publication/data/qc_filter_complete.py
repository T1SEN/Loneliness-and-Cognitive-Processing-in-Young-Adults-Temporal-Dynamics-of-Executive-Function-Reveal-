"""
QC filter for `publication/data/complete/` -> `publication/data/complete/filtered/`.

This script is for cleaning the already "complete" dataset (participants who finished
all tasks). It applies the QC thresholds recorded in `Results.md`, with one robustness
improvement:

- Duration outliers are judged using trial-derived elapsed time
  (`t1_onset_ms` / `stim_onset_ms`), and clearly corrupted `duration_seconds` values in
  `3_cognitive_tests_summary.csv` are automatically corrected in the OUTPUT.

Run:
    python -m publication.data.qc_filter_complete

Outputs (written to `publication/data/complete/filtered/`):
    - Filtered copies of the 6 CSVs
    - `filtered_participant_ids.csv`
    - `excluded_participant_ids.csv`
    - `excluded_participants.csv` (reasons + key metrics, incl. trial QC metrics)
    - `duration_corrections.csv`
    - `qc_summary.json`
    - `excluded_participants_report.md`
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from publication.preprocessing.constants import DEFAULT_RT_MIN, PRP_RT_MAX


if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


COMPLETE_DIR = Path("publication/data/complete")
OUTPUT_DIR = COMPLETE_DIR / "filtered"

SOURCE_FILES = [
    "1_participants_info.csv",
    "2_surveys_results.csv",
    "3_cognitive_tests_summary.csv",
    "4a_prp_trials.csv",
    "4b_wcst_trials.csv",
    "4c_stroop_trials.csv",
]


@dataclass(frozen=True)
class QcThresholds:
    prp_acc_min_pct: float = 80.0
    wcst_total_error_exclude_gte: int = 60
    wcst_persev_resp_exclude_gte: int = 60
    stroop_acc_min_pct: float = 93.0
    stroop_mrt_exclude_gt_ms: float = 1500.0
    trial_duration_exclude_gt_s: float = 1000.0
    duration_correction_abs_diff_s: float = 60.0


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_csv(index=False) + "\n```"


def _trial_duration_seconds(
    prp_trials: pd.DataFrame, stroop_trials: pd.DataFrame, wcst_trials: pd.DataFrame
) -> pd.DataFrame:
    """
    Long table: participantId, testName, trial_duration_s = (max(onset_ms)-min(onset_ms))/1000.
    """
    parts: list[pd.DataFrame] = []

    prp = prp_trials.groupby("participantId")["t1_onset_ms"].agg(["min", "max"]).reset_index()
    prp["testName"] = "prp"
    prp["trial_duration_s"] = (prp["max"] - prp["min"]) / 1000.0
    parts.append(prp[["participantId", "testName", "trial_duration_s"]])

    st = stroop_trials.groupby("participantId")["stim_onset_ms"].agg(["min", "max"]).reset_index()
    st["testName"] = "stroop"
    st["trial_duration_s"] = (st["max"] - st["min"]) / 1000.0
    parts.append(st[["participantId", "testName", "trial_duration_s"]])

    wc = wcst_trials.groupby("participantId")["stim_onset_ms"].agg(["min", "max"]).reset_index()
    wc["testName"] = "wcst"
    wc["trial_duration_s"] = (wc["max"] - wc["min"]) / 1000.0
    parts.append(wc[["participantId", "testName", "trial_duration_s"]])

    out = pd.concat(parts, ignore_index=True)
    out["participantId"] = out["participantId"].astype(str)
    out["testName"] = out["testName"].astype(str).str.lower()
    return out


def _correct_summary_durations(
    summary: pd.DataFrame,
    trial_durations: pd.DataFrame,
    *,
    abs_diff_threshold_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Correct clearly corrupted summary `duration_seconds` values using trial-derived durations.
    """
    summary = summary.copy()
    summary["participantId"] = summary["participantId"].astype(str)
    summary["testName"] = summary["testName"].astype(str).str.lower()

    td = trial_durations.copy()
    td["participantId"] = td["participantId"].astype(str)
    td["testName"] = td["testName"].astype(str).str.lower()

    merged = summary.merge(td, on=["participantId", "testName"], how="left", validate="many_to_one")
    merged["diff_s"] = merged["duration_seconds"] - merged["trial_duration_s"]

    # Per-test robust offset: median(diff) among non-outliers
    offsets: dict[str, float] = {}
    for test_name, g in merged.groupby("testName"):
        g_ok = g[g["diff_s"].abs() <= abs_diff_threshold_s]
        offsets[str(test_name)] = float(np.nanmedian(g_ok["diff_s"])) if len(g_ok) else 0.0

    needs_fix = merged["diff_s"].abs() > abs_diff_threshold_s
    merged["duration_seconds_old"] = merged["duration_seconds"]
    merged.loc[needs_fix, "duration_seconds"] = (
        merged.loc[needs_fix, "trial_duration_s"] + merged.loc[needs_fix, "testName"].map(offsets)
    ).round(3)

    corrections = merged.loc[needs_fix, [
        "participantId",
        "testName",
        "duration_seconds_old",
        "trial_duration_s",
        "duration_seconds",
        "diff_s",
    ]].rename(columns={"duration_seconds": "duration_seconds_new"}).copy()
    corrections = corrections.sort_values(["testName", "participantId"])

    corrected = merged[summary.columns].copy()
    return corrected, corrections


def _build_base_table(info: pd.DataFrame, surveys: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot_table(
        index="participantId",
        columns="testName",
        values=[
            "duration_seconds",
            "acc_t1",
            "acc_t2",
            "mrt_t1",
            "mrt_t2",
            "accuracy",
            "mrt_total",
            "stroop_effect",
            "totalErrorCount",
            "perseverativeResponses",
            "completedCategories",
            "totalTrialCount",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{test}" for metric, test in wide.columns]
    wide = wide.reset_index()

    ucla = (
        surveys[surveys["surveyName"] == "ucla"][["participantId", "duration_seconds", "score"]]
        .rename(columns={"duration_seconds": "ucla_duration_s", "score": "ucla_score"})
        .copy()
    )
    dass = (
        surveys[surveys["surveyName"] == "dass"][["participantId", "duration_seconds", "score_A", "score_S", "score_D"]]
        .rename(
            columns={
                "duration_seconds": "dass_duration_s",
                "score_D": "dass_depression",
                "score_A": "dass_anxiety",
                "score_S": "dass_stress",
            }
        )
        .copy()
    )

    for df in (info, wide, ucla, dass):
        df["participantId"] = df["participantId"].astype(str)

    return info.merge(wide, on="participantId", how="left").merge(ucla, on="participantId", how="left").merge(dass, on="participantId", how="left")


def _compute_exclusions(
    base: pd.DataFrame,
    trial_durations: pd.DataFrame,
    *,
    thresholds: QcThresholds,
) -> tuple[set[str], dict[str, set[str]], dict[str, int]]:
    reasons: dict[str, set[str]] = {}
    rule_counts: dict[str, int] = {}

    def add(pid: str, rule: str) -> None:
        reasons.setdefault(pid, set()).add(rule)

    def record(rule: str, pids: set[str]) -> None:
        rule_counts[rule] = len(pids)
        for pid in pids:
            add(pid, rule)

    record("age<0", set(base.loc[base["age"] < 0, "participantId"]))

    prp_bad = set(
        base.loc[
            (base.get("acc_t1_prp") < thresholds.prp_acc_min_pct)
            | (base.get("acc_t2_prp") < thresholds.prp_acc_min_pct),
            "participantId",
        ].dropna()
    )
    record(f"prp_accuracy<{thresholds.prp_acc_min_pct:.0f}%", prp_bad)

    wcst_bad = set(
        base.loc[
            (base.get("totalErrorCount_wcst") >= thresholds.wcst_total_error_exclude_gte)
            | (base.get("perseverativeResponses_wcst") >= thresholds.wcst_persev_resp_exclude_gte),
            "participantId",
        ].dropna()
    )
    record("wcst_totalError>=60_or_persevResp>=60", wcst_bad)

    stroop_bad = set(
        base.loc[
            (base.get("accuracy_stroop") < thresholds.stroop_acc_min_pct)
            | (base.get("mrt_total_stroop") > thresholds.stroop_mrt_exclude_gt_ms),
            "participantId",
        ].dropna()
    )
    record("stroop_accuracy<93%_or_mrt>1500ms", stroop_bad)

    td_wide = trial_durations.pivot_table(index="participantId", columns="testName", values="trial_duration_s", aggfunc="first")
    duration_bad = set(td_wide[(td_wide > thresholds.trial_duration_exclude_gt_s).any(axis=1)].index.astype(str))
    record(f"trial_duration>{thresholds.trial_duration_exclude_gt_s:.0f}s", duration_bad)

    return set(reasons.keys()), reasons, rule_counts


def _trial_qc_metrics(
    prp_trials: pd.DataFrame,
    stroop_trials: pd.DataFrame,
    wcst_trials: pd.DataFrame,
    participant_ids: set[str],
) -> pd.DataFrame:
    pids = set(map(str, participant_ids))
    out: pd.DataFrame | None = None

    # PRP
    prp = prp_trials[prp_trials["participantId"].astype(str).isin(pids)].copy()
    if len(prp):
        prp["participantId"] = prp["participantId"].astype(str)
        prp["t1_correct"] = prp["t1_correct"].fillna(False).astype(bool)
        prp["t2_correct"] = prp["t2_correct"].fillna(False).astype(bool)
        prp["t1_timeout"] = prp.get("t1_timeout", False)
        prp["t2_timeout"] = prp.get("t2_timeout", False)
        if isinstance(prp["t1_timeout"], pd.Series):
            prp["t1_timeout"] = prp["t1_timeout"].fillna(False).astype(bool)
        if isinstance(prp["t2_timeout"], pd.Series):
            prp["t2_timeout"] = prp["t2_timeout"].fillna(False).astype(bool)

        rt2_col = "t2_rt_ms" if "t2_rt_ms" in prp.columns else "t2_rt"
        prp["rt2_ms"] = pd.to_numeric(prp[rt2_col], errors="coerce")
        prp["valid_t2"] = (
            prp["t1_correct"]
            & prp["t2_correct"]
            & (~prp["t2_timeout"])
            & (prp["rt2_ms"] > DEFAULT_RT_MIN)
            & (prp["rt2_ms"] < PRP_RT_MAX)
        )

        g = prp.groupby("participantId")
        prp_metrics = pd.DataFrame({
            "participantId": g.size().index,
            "prp_n_trials": g.size().values,
            "prp_t2_timeout_n": g["t2_timeout"].sum().astype(int).values,
            "prp_t2_anticipation_n": g.apply(lambda d: int((d["rt2_ms"].notna() & (d["rt2_ms"] <= DEFAULT_RT_MIN)).sum())).values,
            "prp_t2_rt_median_valid_ms": g.apply(lambda d: float(np.nanmedian(d.loc[d["valid_t2"], "rt2_ms"])) if d["valid_t2"].any() else np.nan).values,
            "prp_t1_resp_unique": g["t1_resp"].nunique(dropna=True).values,
            "prp_t2_resp_unique": g["t2_resp"].nunique(dropna=True).values,
        })
        out = prp_metrics

    # Stroop
    st = stroop_trials[stroop_trials["participantId"].astype(str).isin(pids)].copy()
    if len(st):
        st["participantId"] = st["participantId"].astype(str)
        st["correct"] = st["correct"].fillna(False).astype(bool)
        timeout_col = "is_timeout" if "is_timeout" in st.columns else "timeout"
        st["timeout_any"] = st[timeout_col].fillna(False).astype(bool) if timeout_col in st.columns else False
        rt_col = "rt_ms" if "rt_ms" in st.columns else "rt"
        st["rt_ms_num"] = pd.to_numeric(st[rt_col], errors="coerce") if rt_col in st.columns else np.nan

        g = st.groupby("participantId")
        st_metrics = pd.DataFrame({
            "participantId": g.size().index,
            "stroop_n_trials": g.size().values,
            "stroop_timeout_n": g["timeout_any"].sum().astype(int).values,
            "stroop_anticipation_n": g.apply(lambda d: int((d["rt_ms_num"].notna() & (d["rt_ms_num"] <= DEFAULT_RT_MIN)).sum())).values,
            "stroop_acc_pct_trials": (g["correct"].mean().values * 100.0),
            "stroop_rt_median_correct_ms": g.apply(lambda d: float(np.nanmedian(d.loc[d["correct"] & (~d["timeout_any"]), "rt_ms_num"])) if ((d["correct"] & (~d["timeout_any"])).any()) else np.nan).values,
            "stroop_userColor_unique": g["userColor"].nunique(dropna=True).values if "userColor" in st.columns else np.nan,
        })
        out = st_metrics if out is None else out.merge(st_metrics, on="participantId", how="outer")

    # WCST
    wc = wcst_trials[wcst_trials["participantId"].astype(str).isin(pids)].copy()
    if len(wc):
        wc["participantId"] = wc["participantId"].astype(str)
        wc["correct"] = wc["correct"].fillna(False).astype(bool)
        wc["timeout_any"] = wc.get("timeout", False)
        if isinstance(wc["timeout_any"], pd.Series):
            wc["timeout_any"] = wc["timeout_any"].fillna(False).astype(bool)

        rt_col = "rt_ms" if "rt_ms" in wc.columns else ("reactionTimeMs" if "reactionTimeMs" in wc.columns else None)
        wc["rt_ms_num"] = pd.to_numeric(wc[rt_col], errors="coerce") if rt_col else np.nan

        for col in ["isPE", "isNPE", "isPR"]:
            if col in wc.columns:
                wc[col] = wc[col].fillna(False).astype(bool)

        g = wc.groupby("participantId")
        wc_metrics = pd.DataFrame({
            "participantId": g.size().index,
            "wcst_n_trials": g.size().values,
            "wcst_timeout_n": g["timeout_any"].sum().astype(int).values,
            "wcst_anticipation_n": g.apply(lambda d: int((d["rt_ms_num"].notna() & (d["rt_ms_num"] <= DEFAULT_RT_MIN)).sum())).values,
            "wcst_acc_pct_trials": (g["correct"].mean().values * 100.0),
            "wcst_rt_median_ms": g.apply(lambda d: float(np.nanmedian(d.loc[~d["timeout_any"], "rt_ms_num"])) if (~d["timeout_any"]).any() else np.nan).values,
            "wcst_isPE_n": g["isPE"].sum().astype(int).values if "isPE" in wc.columns else np.nan,
            "wcst_isNPE_n": g["isNPE"].sum().astype(int).values if "isNPE" in wc.columns else np.nan,
            "wcst_isPR_n": g["isPR"].sum().astype(int).values if "isPR" in wc.columns else np.nan,
            "wcst_chosenCard_unique": g["chosenCard"].nunique(dropna=True).values if "chosenCard" in wc.columns else np.nan,
        })
        out = wc_metrics if out is None else out.merge(wc_metrics, on="participantId", how="outer")

    return out if out is not None else pd.DataFrame({"participantId": sorted(pids)})


def _render_report(
    *,
    n_total: int,
    included_ids: list[str],
    excluded_ids: list[str],
    base: pd.DataFrame,
    reasons: dict[str, set[str]],
    rule_counts: dict[str, int],
    duration_corrections: pd.DataFrame,
    trial_metrics: pd.DataFrame,
    thresholds: QcThresholds,
) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []

    lines.append("# QC Exclusion Report (complete -> complete/filtered)")
    lines.append("")
    lines.append(f"- Generated at (UTC): {now}")
    lines.append(f"- Input N: {n_total}")
    lines.append(f"- Excluded N: {len(excluded_ids)}")
    lines.append(f"- Included N: {len(included_ids)}")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("- age < 0")
    lines.append(f"- PRP accuracy < {thresholds.prp_acc_min_pct:.0f}%")
    lines.append("- WCST totalErrorCount >= 60 OR perseverativeResponses >= 60")
    lines.append(f"- Stroop accuracy < {thresholds.stroop_acc_min_pct:.0f}% OR mean RT > {thresholds.stroop_mrt_exclude_gt_ms:.0f}ms")
    lines.append(f"- Trial-derived duration > {thresholds.trial_duration_exclude_gt_s:.0f}s")
    lines.append("")
    lines.append("## Counts (per rule, not unique)")
    for k, v in rule_counts.items():
        lines.append(f"- {k}: {v}")

    if len(duration_corrections):
        lines.append("")
        lines.append("## Duration Corrections (summary duration was corrupted)")
        lines.append("")
        lines.append(_safe_markdown_table(duration_corrections))

    # Overview table
    excluded_df = base[base["participantId"].astype(str).isin(excluded_ids)].copy()
    excluded_df["reasons"] = excluded_df["participantId"].astype(str).map(lambda pid: ";".join(sorted(reasons.get(str(pid), set()))))
    overview_cols = [
        "participantId",
        "gender",
        "age",
        "ucla_score",
        "dass_depression",
        "dass_anxiety",
        "dass_stress",
        "acc_t1_prp",
        "acc_t2_prp",
        "accuracy_stroop",
        "mrt_total_stroop",
        "totalErrorCount_wcst",
        "perseverativeResponses_wcst",
        "reasons",
    ]
    overview_cols = [c for c in overview_cols if c in excluded_df.columns]
    excluded_df = excluded_df[overview_cols].sort_values("participantId")

    lines.append("")
    lines.append("## Excluded Participants (overview)")
    lines.append("")
    lines.append(_safe_markdown_table(excluded_df))

    # Deep dive
    base_idx = base.set_index(base["participantId"].astype(str))
    trial_idx = trial_metrics.set_index(trial_metrics["participantId"].astype(str)) if len(trial_metrics) else pd.DataFrame().set_index([])

    lines.append("")
    lines.append("## Per-Participant Details")
    for pid in excluded_ids:
        row = base_idx.loc[str(pid)]
        tm = trial_idx.loc[str(pid)] if str(pid) in trial_idx.index else None

        lines.append("")
        lines.append(f"### {pid}")
        lines.append(f"- Reasons: {'; '.join(sorted(reasons.get(str(pid), set())))}")

        demo_keys = ["gender", "age", "birthDate", "education", "createdAt"]
        demo = [f"{k}={row[k]}" for k in demo_keys if (k in row.index and not pd.isna(row[k]))]
        if demo:
            lines.append(f"- Demographics: {', '.join(demo)}")

        surv_keys = ["ucla_score", "ucla_duration_s", "dass_depression", "dass_anxiety", "dass_stress", "dass_duration_s"]
        surv = [f"{k}={row[k]}" for k in surv_keys if (k in row.index and not pd.isna(row[k]))]
        if surv:
            lines.append(f"- Surveys: {', '.join(surv)}")

        summ_keys = [
            "duration_seconds_prp",
            "acc_t1_prp",
            "acc_t2_prp",
            "duration_seconds_stroop",
            "accuracy_stroop",
            "mrt_total_stroop",
            "stroop_effect_stroop",
            "duration_seconds_wcst",
            "totalTrialCount_wcst",
            "completedCategories_wcst",
            "totalErrorCount_wcst",
            "perseverativeResponses_wcst",
        ]
        summ = [f"{k}={row[k]}" for k in summ_keys if (k in row.index and not pd.isna(row[k]))]
        if summ:
            lines.append(f"- Summary: {', '.join(summ)}")

        if tm is not None:
            tm_keys = [
                "prp_n_trials",
                "prp_t2_timeout_n",
                "prp_t2_anticipation_n",
                "prp_t2_rt_median_valid_ms",
                "prp_t1_resp_unique",
                "prp_t2_resp_unique",
                "stroop_n_trials",
                "stroop_timeout_n",
                "stroop_anticipation_n",
                "stroop_acc_pct_trials",
                "stroop_rt_median_correct_ms",
                "stroop_userColor_unique",
                "wcst_n_trials",
                "wcst_timeout_n",
                "wcst_anticipation_n",
                "wcst_acc_pct_trials",
                "wcst_rt_median_ms",
                "wcst_isPE_n",
                "wcst_isNPE_n",
                "wcst_isPR_n",
                "wcst_chosenCard_unique",
            ]
            tm_vals = [f"{k}={tm[k]}" for k in tm_keys if (k in tm.index and not pd.isna(tm[k]))]
            if tm_vals:
                lines.append(f"- Trial QC: {', '.join(tm_vals)}")

    return "\n".join(lines) + "\n"


def main() -> None:
    thresholds = QcThresholds()

    info = _read_csv(COMPLETE_DIR / "1_participants_info.csv")
    surveys = _read_csv(COMPLETE_DIR / "2_surveys_results.csv")
    summary_raw = _read_csv(COMPLETE_DIR / "3_cognitive_tests_summary.csv")
    prp_trials = _read_csv(COMPLETE_DIR / "4a_prp_trials.csv")
    wcst_trials = _read_csv(COMPLETE_DIR / "4b_wcst_trials.csv")
    stroop_trials = _read_csv(COMPLETE_DIR / "4c_stroop_trials.csv")

    trial_durations = _trial_duration_seconds(prp_trials, stroop_trials, wcst_trials)
    summary, duration_corrections = _correct_summary_durations(
        summary_raw,
        trial_durations,
        abs_diff_threshold_s=thresholds.duration_correction_abs_diff_s,
    )

    base = _build_base_table(info, surveys, summary)
    excluded_set, reasons, rule_counts = _compute_exclusions(base, trial_durations, thresholds=thresholds)

    excluded_ids = sorted(excluded_set)
    included_ids = sorted(set(base["participantId"].astype(str)) - excluded_set)

    print(f"[QC] Input participants: {len(base)}")
    print(f"[QC] Excluded participants: {len(excluded_ids)}")
    print(f"[QC] Included participants: {len(included_ids)}")

    trial_metrics = _trial_qc_metrics(prp_trials, stroop_trials, wcst_trials, excluded_set)

    # Save metadata
    _write_csv(pd.DataFrame({"participantId": included_ids}), OUTPUT_DIR / "filtered_participant_ids.csv")
    _write_csv(pd.DataFrame({"participantId": excluded_ids}), OUTPUT_DIR / "excluded_participant_ids.csv")

    excluded_df = base[base["participantId"].astype(str).isin(excluded_ids)].copy()
    excluded_df["reasons"] = excluded_df["participantId"].astype(str).map(lambda pid: ";".join(sorted(reasons.get(str(pid), set()))))
    excluded_df = excluded_df.merge(trial_metrics, on="participantId", how="left")
    _write_csv(excluded_df, OUTPUT_DIR / "excluded_participants.csv")

    _write_csv(duration_corrections, OUTPUT_DIR / "duration_corrections.csv")

    qc_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_n": int(len(base)),
        "excluded_n": int(len(excluded_ids)),
        "included_n": int(len(included_ids)),
        "thresholds": {
            "age_exclude_lt": 0,
            "prp_accuracy_exclude_lt_pct": thresholds.prp_acc_min_pct,
            "wcst_total_error_exclude_gte": thresholds.wcst_total_error_exclude_gte,
            "wcst_persev_resp_exclude_gte": thresholds.wcst_persev_resp_exclude_gte,
            "stroop_accuracy_exclude_lt_pct": thresholds.stroop_acc_min_pct,
            "stroop_mrt_exclude_gt_ms": thresholds.stroop_mrt_exclude_gt_ms,
            "trial_duration_exclude_gt_s": thresholds.trial_duration_exclude_gt_s,
            "duration_correction_abs_diff_s": thresholds.duration_correction_abs_diff_s,
        },
        "rule_counts_not_unique": rule_counts,
        "duration_corrections_n": int(len(duration_corrections)),
    }
    _write_json(qc_summary, OUTPUT_DIR / "qc_summary.json")

    report = _render_report(
        n_total=len(base),
        included_ids=included_ids,
        excluded_ids=excluded_ids,
        base=base,
        reasons=reasons,
        rule_counts=rule_counts,
        duration_corrections=duration_corrections,
        trial_metrics=trial_metrics,
        thresholds=thresholds,
    )
    _write_text(report, OUTPUT_DIR / "excluded_participants_report.md")

    # Write filtered datasets
    keep = set(included_ids)
    for filename in SOURCE_FILES:
        src = COMPLETE_DIR / filename
        dst = OUTPUT_DIR / filename

        if filename == "3_cognitive_tests_summary.csv":
            df = summary
        else:
            df = _read_csv(src)

        if "participantId" not in df.columns:
            raise KeyError(f"{filename} missing participantId column")

        df_f = df[df["participantId"].astype(str).isin(keep)].copy()
        _write_csv(df_f, dst)

    print(f"[QC] Wrote filtered dataset to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""Analyze modify trials as confidence-update events in the truss data.

This script treats modify trials as a mechanism in the confidence dynamics,
rather than as a direct performance category. It focuses on:

- modify x positive/negative feedback effects on AI/self confidence deltas
- participant-level robustness of those effects
- next-trial behavior after positive/negative modify trials
- subtype sample sizes and participant coverage
- aligned vs non-aligned sample sizes
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
TRIAL_PATH = ROOT / "trial_level_modify_analysis.csv"
OUTPUT_DIR = ROOT


def _safe_ttest(a: pd.Series, b: pd.Series, *, paired: bool = False) -> Dict[str, float]:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if paired:
        n = min(len(a), len(b))
        if n < 2:
            return {"statistic": np.nan, "p_value": np.nan}
        result = stats.ttest_rel(a.iloc[:n], b.iloc[:n])
    else:
        if len(a) < 2 or len(b) < 2:
            return {"statistic": np.nan, "p_value": np.nan}
        result = stats.ttest_ind(a, b, equal_var=False)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def clustered_ols_positive_effect(data: pd.DataFrame, outcome: str) -> Dict[str, float]:
    """OLS outcome ~ positive_feedback + condition2 + trial with cluster SE.

    Returns the positive_feedback coefficient, t statistic, and normal-approx p.
    This avoids requiring statsmodels for a small diagnostic regression.
    """
    cols = [outcome, "positive_feedback", "condition", "trial", "participant"]
    work = data[cols].dropna().reset_index(drop=True)
    y = work[outcome].to_numpy(dtype=float)
    x = np.column_stack([
        np.ones(work.shape[0], dtype=float),
        work["positive_feedback"].to_numpy(dtype=float),
        (work["condition"].to_numpy(dtype=float) == 2).astype(float),
        work["trial"].to_numpy(dtype=float),
    ])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)

    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for _, idx in work.groupby("participant").groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        xg = x[idx_arr, :]
        ug = residual[idx_arr]
        xu = xg.T @ ug
        meat += np.outer(xu, xu)

    n = x.shape[0]
    k = x.shape[1]
    clusters = work["participant"].nunique()
    if clusters > 1 and n > k:
        correction = (clusters / (clusters - 1)) * ((n - 1) / (n - k))
    else:
        correction = 1.0
    cov = correction * xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    coef = float(beta[1])
    t_value = coef / se if se > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(t_value))) if np.isfinite(t_value) else np.nan
    return {"estimate": coef, "statistic": float(t_value), "p_value": p_value}


def confidence_effects(df: pd.DataFrame) -> None:
    modify = df[df["action"] == "modify"].copy()

    summary_rows: List[Dict[str, object]] = []
    for feedback, group in modify.groupby("positive_feedback", sort=True):
        label = "positive_modify" if int(feedback) == 1 else "negative_modify"
        row: Dict[str, object] = {
            "feedback_group": label,
            "positive_feedback": int(feedback),
            "n_trials": int(group.shape[0]),
            "n_participants": int(group["participant"].nunique()),
            "condition1_trials": int((group["condition"] == 1).sum()),
            "condition2_trials": int((group["condition"] == 2).sum()),
        }
        for col in ["delta_ai", "delta_self", "delta_gap_self_minus_ai"]:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=1))
            row[f"{col}_sem"] = float(values.sem())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "modify_feedback_confidence_delta_summary.csv",
        index=False,
    )

    positive = modify[modify["positive_feedback"] == 1]
    negative = modify[modify["positive_feedback"] == 0]

    test_rows = []
    for outcome in ["delta_ai", "delta_self", "delta_gap_self_minus_ai"]:
        trial_test = _safe_ttest(positive[outcome], negative[outcome])
        test_rows.append({
            "outcome": outcome,
            "test": "trial_level_welch_positive_vs_negative_modify",
            "n_positive": int(positive[outcome].notna().sum()),
            "n_negative": int(negative[outcome].notna().sum()),
            "estimate_positive_minus_negative": float(positive[outcome].mean() - negative[outcome].mean()),
            **trial_test,
        })

        participant_means = (
            modify
            .groupby(["participant", "positive_feedback"], as_index=False)[outcome]
            .mean()
            .pivot(index="participant", columns="positive_feedback", values=outcome)
            .dropna()
        )
        if {0, 1}.issubset(set(participant_means.columns)):
            paired_test = _safe_ttest(participant_means[1], participant_means[0], paired=True)
            estimate = float((participant_means[1] - participant_means[0]).mean())
            n_pair = int(participant_means.shape[0])
        else:
            paired_test = {"statistic": np.nan, "p_value": np.nan}
            estimate = np.nan
            n_pair = 0
        test_rows.append({
            "outcome": outcome,
            "test": "participant_paired_positive_vs_negative_modify",
            "n_positive": n_pair,
            "n_negative": n_pair,
            "estimate_positive_minus_negative": estimate,
            **paired_test,
        })

        regression = clustered_ols_positive_effect(modify, outcome)
        test_rows.append({
            "outcome": outcome,
            "test": "clustered_ols_positive_feedback_controlling_condition_trial",
            "n_positive": int(positive[outcome].notna().sum()),
            "n_negative": int(negative[outcome].notna().sum()),
            "estimate_positive_minus_negative": regression["estimate"],
            "statistic": regression["statistic"],
            "p_value": regression["p_value"],
        })

    pd.DataFrame(test_rows).to_csv(
        OUTPUT_DIR / "modify_feedback_confidence_delta_tests.csv",
        index=False,
    )


def next_behavior(df: pd.DataFrame) -> None:
    ordered = df.sort_values(["participant", "condition", "trial"]).copy()
    grouped = ordered.groupby(["participant", "condition"], sort=False)
    ordered["next_action"] = grouped["action"].shift(-1)
    ordered["next_positive_feedback"] = grouped["positive_feedback"].shift(-1)
    ordered["next_delta_ai"] = grouped["delta_ai"].shift(-1)
    ordered["next_delta_self"] = grouped["delta_self"].shift(-1)

    modify_next = ordered[(ordered["action"] == "modify") & ordered["next_action"].notna()].copy()

    counts = (
        modify_next
        .groupby(["positive_feedback", "next_action"])
        .size()
        .reset_index(name="n")
    )
    totals = counts.groupby("positive_feedback")["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    counts.to_csv(OUTPUT_DIR / "modify_feedback_next_action_counts.csv", index=False)

    ctab = pd.crosstab(modify_next["positive_feedback"], modify_next["next_action"])
    chi2, p_value, dof, _ = stats.chi2_contingency(ctab)
    pd.DataFrame([{
        "test": "chi_square_modify_feedback_by_next_action",
        "chi2": float(chi2),
        "p_value": float(p_value),
        "df": int(dof),
        "n": int(ctab.to_numpy().sum()),
    }]).to_csv(OUTPUT_DIR / "modify_feedback_next_action_tests.csv", index=False)

    next_delta = (
        modify_next
        .groupby("positive_feedback")
        .agg(
            n=("participant", "size"),
            next_positive_feedback_rate=("next_positive_feedback", "mean"),
            next_delta_ai_mean=("next_delta_ai", "mean"),
            next_delta_self_mean=("next_delta_self", "mean"),
        )
        .reset_index()
    )
    next_delta.to_csv(OUTPUT_DIR / "modify_feedback_next_trial_summary.csv", index=False)


def subtype_robustness(df: pd.DataFrame) -> None:
    modify = df[df["action"] == "modify"].copy()
    rows = []
    specs = [
        ("outcome", "modify_outcome_label"),
        ("agency", "modify_agency_label"),
        ("coupling", "modify_coupling_label"),
    ]
    for dimension, col in specs:
        labeled = modify[modify[col].notna() & (modify[col] != "")]
        for label, group in labeled.groupby(col, sort=True):
            per_participant = group.groupby("participant").size()
            rows.append({
                "dimension": dimension,
                "label": label,
                "n_trials": int(group.shape[0]),
                "share_of_modify": float(group.shape[0] / modify.shape[0]),
                "n_participants": int(group["participant"].nunique()),
                "condition1_trials": int((group["condition"] == 1).sum()),
                "condition2_trials": int((group["condition"] == 2).sum()),
                "median_trials_per_participant_with_label": float(per_participant.median()),
                "max_trials_single_participant": int(per_participant.max()),
                "positive_feedback_rate": float(group["positive_feedback"].mean()),
                "delta_ai_mean": float(group["delta_ai"].mean()),
                "delta_self_mean": float(group["delta_self"].mean()),
            })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "modify_subtype_robustness_audit.csv", index=False)


def alignment_counts(df: pd.DataFrame) -> None:
    rows = []
    for align_value, group in df.groupby("align", sort=True):
        align_label = "aligned" if float(align_value) == 1.0 else "non_aligned"
        action_counts = group["action"].value_counts()
        rows.append({
            "align_group": align_label,
            "n_trials": int(group.shape[0]),
            "n_participants": int(group["participant"].nunique()),
            "accept_ai_trials": int(action_counts.get("accept_ai", 0)),
            "reject_ai_trials": int(action_counts.get("reject_ai", 0)),
            "modify_trials": int(action_counts.get("modify", 0)),
            "accept_ai_rate": float(action_counts.get("accept_ai", 0) / group.shape[0]),
            "reject_ai_rate": float(action_counts.get("reject_ai", 0) / group.shape[0]),
            "modify_rate": float(action_counts.get("modify", 0) / group.shape[0]),
            "positive_feedback_rate": float(group["positive_feedback"].mean()),
        })

    non_aligned = df[df["align"] == 0]
    for action, group in non_aligned.groupby("action", sort=True):
        rows.append({
            "align_group": f"non_aligned_{action}",
            "n_trials": int(group.shape[0]),
            "n_participants": int(group["participant"].nunique()),
            "accept_ai_trials": int((group["action"] == "accept_ai").sum()),
            "reject_ai_trials": int((group["action"] == "reject_ai").sum()),
            "modify_trials": int((group["action"] == "modify").sum()),
            "accept_ai_rate": float((group["action"] == "accept_ai").mean()),
            "reject_ai_rate": float((group["action"] == "reject_ai").mean()),
            "modify_rate": float((group["action"] == "modify").mean()),
            "positive_feedback_rate": float(group["positive_feedback"].mean()),
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "alignment_nonaligned_sample_audit.csv", index=False)


def confidence_coupling(df: pd.DataFrame) -> None:
    rows = []
    for label, group in [
        ("all_trials", df),
        ("modify_trials", df[df["action"] == "modify"]),
        ("non_modify_trials", df[df["action"] != "modify"]),
        ("non_aligned_trials", df[df["align"] == 0]),
        ("non_aligned_modify_trials", df[(df["align"] == 0) & (df["action"] == "modify")]),
    ]:
        valid = group[["delta_ai", "delta_self"]].dropna()
        if valid.shape[0] >= 3:
            pearson = stats.pearsonr(valid["delta_ai"], valid["delta_self"])
            spearman = stats.spearmanr(valid["delta_ai"], valid["delta_self"])
            rows.append({
                "sample": label,
                "n": int(valid.shape[0]),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "same_direction_rate": float(((valid["delta_ai"] * valid["delta_self"]) > 0).mean()),
                "opposite_direction_rate": float(((valid["delta_ai"] * valid["delta_self"]) < 0).mean()),
                "zero_or_mixed_rate": float(((valid["delta_ai"] * valid["delta_self"]) == 0).mean()),
            })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "confidence_channel_coupling_summary.csv", index=False)


def confidence_quadrant_contexts(df: pd.DataFrame) -> None:
    samples = [
        ("all_trials", df),
        ("modify_trials", df[df["action"] == "modify"]),
        ("non_aligned_trials", df[df["align"] == 0]),
        ("non_aligned_modify_trials", df[(df["align"] == 0) & (df["action"] == "modify")]),
    ]
    quadrant_order = [
        "AI_up_self_up",
        "AI_down_self_down",
        "AI_up_self_down",
        "AI_down_self_up",
        "flat_or_mixed_zero",
    ]
    rows = []
    for sample_name, sample in samples:
        total = sample.shape[0]
        for quadrant in quadrant_order:
            group = sample[sample["quadrant"] == quadrant]
            if group.empty:
                continue
            action_rates = group["action"].value_counts(normalize=True)
            rows.append({
                "sample": sample_name,
                "quadrant": quadrant,
                "n": int(group.shape[0]),
                "share_of_sample": float(group.shape[0] / total) if total else np.nan,
                "n_participants": int(group["participant"].nunique()),
                "positive_feedback_rate": float(group["positive_feedback"].mean()),
                "condition2_rate": float((group["condition"] == 2).mean()),
                "align_rate": float(group["align"].mean()),
                "accept_ai_rate": float(action_rates.get("accept_ai", 0.0)),
                "reject_ai_rate": float(action_rates.get("reject_ai", 0.0)),
                "modify_rate": float(action_rates.get("modify", 0.0)),
                "ai_before_mean": float(group["ai_before"].mean()),
                "self_before_mean": float(group["self_before"].mean()),
                "gap_self_minus_ai_before_mean": float((group["self_before"] - group["ai_before"]).mean()),
                "delta_ai_mean": float(group["delta_ai"].mean()),
                "delta_self_mean": float(group["delta_self"].mean()),
                "team_score_mean": float(group["team_score"].mean()),
                "individual_score_mean": float(group["individual_score"].mean()),
            })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "confidence_quadrant_context_summary.csv", index=False)


def main() -> None:
    df = pd.read_csv(TRIAL_PATH)
    confidence_effects(df)
    next_behavior(df)
    subtype_robustness(df)
    alignment_counts(df)
    confidence_coupling(df)
    confidence_quadrant_contexts(df)
    print(f"Saved modify confidence-role outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

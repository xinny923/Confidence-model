from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "modify_analysis"
RECON_PATH = ROOT / "data_folder_output" / "data_folder_reconstructed_trials.csv"
RANDOM_SEED = 42
N_PERMUTATIONS = 5000
EPS = 1e-12

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import truss_detailed_subset_fit as detailed  # noqa: E402
from truss_og_benchmark import _cell_to_conditions, _load_first_variable  # noqa: E402


def score_lookup() -> dict[tuple[int, int], tuple[float, float]]:
    score1 = _cell_to_conditions(_load_first_variable(ROOT / "score1_data.mat"), 2, "score1_data")
    score2 = _cell_to_conditions(_load_first_variable(ROOT / "score2_data.mat"), 2, "score2_data")
    scores: dict[tuple[int, int], tuple[float, float]] = {}
    for cond in (1, 2):
        individual = np.ravel(score1[cond]).astype(float)
        team = np.ravel(score2[cond]).astype(float)
        for condition_row, (ind_score, team_score) in enumerate(zip(individual, team)):
            scores[(cond, condition_row)] = (float(ind_score), float(team_score))
    return scores


def confidence_delta_label(delta_ai: float, delta_self: float, positive_feedback: bool) -> tuple[str, str, str]:
    outcome_label = "productive_modify" if positive_feedback else "unproductive_modify"

    if delta_self > EPS and delta_ai <= EPS:
        agency_label = "self_correction"
    elif delta_ai > EPS and delta_self >= -EPS:
        agency_label = "partial_acceptance"
    else:
        agency_label = "agency_unclear"

    if positive_feedback and delta_ai > EPS and delta_self > EPS:
        coupling_label = "collaborative_refinement"
    elif delta_ai * delta_self < -EPS:
        coupling_label = "conflict_modify"
    else:
        coupling_label = "coupling_unclear"

    return outcome_label, agency_label, coupling_label


def quadrant(delta_ai: float, delta_self: float) -> str:
    if delta_ai > EPS and delta_self > EPS:
        return "AI_up_self_up"
    if delta_ai > EPS and delta_self < -EPS:
        return "AI_up_self_down"
    if delta_ai < -EPS and delta_self > EPS:
        return "AI_down_self_up"
    if delta_ai < -EPS and delta_self < -EPS:
        return "AI_down_self_down"
    return "flat_or_mixed_zero"


def build_trial_table() -> pd.DataFrame:
    matrices, metadata = detailed.build_subset_matrices()
    recon = pd.read_csv(RECON_PATH)
    scores = score_lookup()
    meta = metadata.set_index("participant")
    recon_keyed = {
        (int(row.participant), int(row.trial)): row
        for row in recon.itertuples(index=False)
    }

    rows = []
    for idx, pid in enumerate(matrices["participant_array"].astype(int)):
        cond = int(matrices["condition_array"][idx])
        condition_row = int(meta.loc[pid, "condition_row"])
        individual_score, team_score = scores[(cond, condition_row)]

        for t in range(30):
            rec = recon_keyed[(pid, t + 1)]
            action = str(rec.reconstructed_action)
            positive_feedback = int(np.nanargmax(matrices["baseline_e_tensor"][idx, t, :])) in (0, 1)
            ai_before = float(matrices["ai_conf_matrix"][idx, t])
            ai_after = float(matrices["ai_conf_matrix"][idx, t + 1])
            self_before = float(matrices["self_conf_matrix"][idx, t])
            self_after = float(matrices["self_conf_matrix"][idx, t + 1])
            delta_ai = ai_after - ai_before
            delta_self = self_after - self_before

            outcome_label = ""
            agency_label = ""
            coupling_label = ""
            if action == "modify":
                outcome_label, agency_label, coupling_label = confidence_delta_label(
                    delta_ai, delta_self, positive_feedback
                )

            rows.append(
                {
                    "participant": pid,
                    "condition": cond,
                    "condition_row": condition_row,
                    "trial": t + 1,
                    "action": action,
                    "align": float(rec.align),
                    "positive_feedback": int(positive_feedback),
                    "ai_before": ai_before,
                    "ai_after": ai_after,
                    "self_before": self_before,
                    "self_after": self_after,
                    "delta_ai": delta_ai,
                    "delta_self": delta_self,
                    "delta_gap_self_minus_ai": delta_self - delta_ai,
                    "quadrant": quadrant(delta_ai, delta_self),
                    "modify_outcome_label": outcome_label,
                    "modify_agency_label": agency_label,
                    "modify_coupling_label": coupling_label,
                    "individual_score": individual_score,
                    "team_score": team_score,
                }
            )
    return pd.DataFrame(rows)


def rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def summarize_action_feedback(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in trials.groupby(["condition", "action"], dropna=False):
        condition, action = keys
        rows.append(
            {
                "condition": condition,
                "action": action,
                "n": len(grp),
                "positive_feedback_rate": rate(grp["positive_feedback"]),
                "mean_delta_ai": grp["delta_ai"].mean(),
                "mean_delta_self": grp["delta_self"].mean(),
                "mean_delta_gap_self_minus_ai": grp["delta_gap_self_minus_ai"].mean(),
            }
        )
    rows.append(
        {
            "condition": "all",
            "action": "all",
            "n": len(trials),
            "positive_feedback_rate": rate(trials["positive_feedback"]),
            "mean_delta_ai": trials["delta_ai"].mean(),
            "mean_delta_self": trials["delta_self"].mean(),
            "mean_delta_gap_self_minus_ai": trials["delta_gap_self_minus_ai"].mean(),
        }
    )
    return pd.DataFrame(rows)


def summarize_confidence_deltas(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action, grp in trials.groupby("action"):
        for quad, qgrp in grp.groupby("quadrant"):
            rows.append(
                {
                    "action": action,
                    "quadrant": quad,
                    "n": len(qgrp),
                    "share_within_action": len(qgrp) / len(grp),
                    "positive_feedback_rate": rate(qgrp["positive_feedback"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["action", "quadrant"])


def action_feedback_table(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action in ["accept_ai", "reject_ai", "modify"]:
        grp = trials[trials["action"] == action]
        rows.append(
            {
                "action": action,
                "n": len(grp),
                "positive_percent": 100.0 * rate(grp["positive_feedback"]),
                "negative_percent": 100.0 * (1.0 - rate(grp["positive_feedback"])),
            }
        )
    return pd.DataFrame(rows)


def action_confidence_quadrant_table(trials: pd.DataFrame) -> pd.DataFrame:
    quadrant_map = {
        "AI_up_self_up": "+/+",
        "AI_up_self_down": "+/-",
        "AI_down_self_up": "-/+",
        "AI_down_self_down": "-/-",
    }
    rows = []
    for action in ["accept_ai", "reject_ai", "modify"]:
        grp = trials[trials["action"] == action]
        nonzero = grp[grp["quadrant"].isin(quadrant_map)]
        row = {"action": action, "nonzero_n": len(nonzero)}
        for source, label in quadrant_map.items():
            row[label] = int((grp["quadrant"] == source).sum())
            row[f"{label}_percent_of_action"] = 100.0 * row[label] / len(grp) if len(grp) else np.nan
            row[f"{label}_percent_of_nonzero"] = 100.0 * row[label] / len(nonzero) if len(nonzero) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_modify_labels(trials: pd.DataFrame) -> pd.DataFrame:
    modify = trials[trials["action"] == "modify"].copy()
    rows = []
    dimensions = [
        ("outcome", "modify_outcome_label"),
        ("agency", "modify_agency_label"),
        ("coupling", "modify_coupling_label"),
    ]
    for dimension, col in dimensions:
        for label, grp in modify.groupby(col):
            rows.append(
                {
                    "dimension": dimension,
                    "label": label,
                    "n": len(grp),
                    "share_of_modify": len(grp) / len(modify),
                    "positive_feedback_rate": rate(grp["positive_feedback"]),
                    "mean_delta_ai": grp["delta_ai"].mean(),
                    "mean_delta_self": grp["delta_self"].mean(),
                    "mean_team_score": grp["team_score"].mean(),
                    "mean_individual_score": grp["individual_score"].mean(),
                }
            )
    return pd.DataFrame(rows)


def participant_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, grp in trials.groupby("participant"):
        modify = grp[grp["action"] == "modify"]
        productive = modify[modify["modify_outcome_label"] == "productive_modify"]
        collaborative = modify[modify["modify_coupling_label"] == "collaborative_refinement"]
        conflict = modify[modify["modify_coupling_label"] == "conflict_modify"]
        self_correction = modify[modify["modify_agency_label"] == "self_correction"]
        partial_acceptance = modify[modify["modify_agency_label"] == "partial_acceptance"]
        rows.append(
            {
                "participant": pid,
                "condition": int(grp["condition"].iloc[0]),
                "individual_score": float(grp["individual_score"].iloc[0]),
                "team_score": float(grp["team_score"].iloc[0]),
                "accept_rate": rate(grp["action"].eq("accept_ai")),
                "reject_rate": rate(grp["action"].eq("reject_ai")),
                "modify_rate": rate(grp["action"].eq("modify")),
                "align_rate": rate(grp["align"]),
                "positive_feedback_rate": rate(grp["positive_feedback"]),
                "positive_modify_rate": len(productive) / len(grp),
                "productive_modify_ratio": len(productive) / len(modify) if len(modify) else np.nan,
                "collaborative_modify_rate": len(collaborative) / len(grp),
                "collaborative_modify_ratio": len(collaborative) / len(modify) if len(modify) else np.nan,
                "conflict_modify_rate": len(conflict) / len(grp),
                "self_correction_modify_rate": len(self_correction) / len(grp),
                "partial_acceptance_modify_rate": len(partial_acceptance) / len(grp),
                "mean_delta_ai_after_modify": modify["delta_ai"].mean(),
                "mean_delta_self_after_modify": modify["delta_self"].mean(),
                "mean_ai_confidence": grp[["ai_before", "ai_after"]].to_numpy().mean(),
                "mean_self_confidence": grp[["self_before", "self_after"]].to_numpy().mean(),
                "n_modify": len(modify),
            }
        )
    return pd.DataFrame(rows)


def performer_summary(participants: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accept_rate",
        "modify_rate",
        "positive_modify_rate",
        "productive_modify_ratio",
        "collaborative_modify_rate",
        "collaborative_modify_ratio",
        "conflict_modify_rate",
        "self_correction_modify_rate",
        "partial_acceptance_modify_rate",
        "positive_feedback_rate",
    ]
    rows = []
    for score_col in ["team_score", "individual_score"]:
        median = participants[score_col].median()
        labeled = participants.copy()
        labeled["performer_group"] = np.where(labeled[score_col] >= median, "high", "low")
        for group, grp in labeled.groupby("performer_group"):
            row = {
                "score_basis": score_col,
                "median_cut": median,
                "performer_group": group,
                "n": len(grp),
                "mean_score": grp[score_col].mean(),
            }
            for metric in metrics:
                row[metric] = grp[metric].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def participant_correlations(participants: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accept_rate",
        "modify_rate",
        "positive_modify_rate",
        "productive_modify_ratio",
        "collaborative_modify_rate",
        "collaborative_modify_ratio",
        "conflict_modify_rate",
        "self_correction_modify_rate",
        "partial_acceptance_modify_rate",
        "mean_delta_ai_after_modify",
        "mean_delta_self_after_modify",
    ]
    rows = []
    for score_col in ["team_score", "individual_score"]:
        for metric in metrics:
            sub = participants[[score_col, metric]].dropna()
            if len(sub) < 3 or sub[metric].nunique() < 2:
                continue
            pearson = stats.pearsonr(sub[metric], sub[score_col])
            spearman = stats.spearmanr(sub[metric], sub[score_col])
            rows.append(
                {
                    "score": score_col,
                    "metric": metric,
                    "n": len(sub),
                    "pearson_r": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "spearman_r": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                }
            )
    return pd.DataFrame(rows)


def ols_table(participants: pd.DataFrame, outcome: str) -> pd.DataFrame:
    predictors = [
        "condition_2",
        "accept_rate",
        "modify_rate",
        "positive_modify_rate",
        "collaborative_modify_rate",
        "conflict_modify_rate",
        "mean_ai_confidence",
        "mean_self_confidence",
    ]
    data = participants.copy()
    data["condition_2"] = (data["condition"] == 2).astype(float)
    data = data[[outcome, *predictors]].dropna()
    y = data[outcome].to_numpy(dtype=float)
    x = data[predictors].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    names = ["intercept", *predictors]

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    resid = y - fitted
    df = len(y) - x.shape[1]
    sigma2 = float((resid @ resid) / df)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    t_values = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_values), df)
    r2 = 1.0 - float(np.sum(resid**2) / np.sum((y - y.mean()) ** 2))
    adj_r2 = 1.0 - (1.0 - r2) * (len(y) - 1) / df

    rng = np.random.default_rng(RANDOM_SEED)
    perm_counts = np.zeros_like(beta)
    observed_abs = np.abs(beta)
    for _ in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y)
        beta_perm, *_ = np.linalg.lstsq(x, y_perm, rcond=None)
        perm_counts += (np.abs(beta_perm) >= observed_abs).astype(int)
    perm_p = (perm_counts + 1) / (N_PERMUTATIONS + 1)

    return pd.DataFrame(
        {
            "outcome": outcome,
            "term": names,
            "coef": beta,
            "std_error": se,
            "t": t_values,
            "p": p_values,
            "permutation_p": perm_p,
            "n": len(y),
            "r2": r2,
            "adj_r2": adj_r2,
        }
    )


def save_plots(trials: pd.DataFrame, participants: pd.DataFrame) -> None:
    action_summary = (
        trials.groupby("action")["positive_feedback"]
        .mean()
        .reindex(["accept_ai", "modify", "reject_ai"])
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    action_summary.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylabel("Positive feedback rate")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.set_title("Outcome by action")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "action_positive_feedback_rates.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    colors = {"accept_ai": "#4c78a8", "modify": "#f58518", "reject_ai": "#54a24b"}
    for action, grp in trials.groupby("action"):
        ax.scatter(grp["delta_ai"], grp["delta_self"], s=14, alpha=0.42, label=action, color=colors.get(action))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Delta AI confidence")
    ax.set_ylabel("Delta self confidence")
    ax.set_title("Conjugate confidence dynamics")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confidence_delta_quadrants.png", dpi=220)
    plt.close(fig)

    label_counts = summarize_modify_labels(trials)
    label_order = {
        "outcome": ["productive_modify", "unproductive_modify"],
        "agency": ["self_correction", "partial_acceptance", "agency_unclear"],
        "coupling": ["collaborative_refinement", "conflict_modify", "coupling_unclear"],
    }
    label_display = {
        "productive_modify": "positive-outcome",
        "unproductive_modify": "negative-outcome",
        "self_correction": "self-correction\nsignature",
        "partial_acceptance": "partial-acceptance\nsignature",
        "agency_unclear": "agency unclear",
        "collaborative_refinement": "collaborative-refinement\nsignature",
        "conflict_modify": "conflict\nsignature",
        "coupling_unclear": "coupling unclear",
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), sharey=True)
    colors = {
        "productive_modify": "#54a24b",
        "unproductive_modify": "#e45756",
        "self_correction": "#72b7b2",
        "partial_acceptance": "#4c78a8",
        "agency_unclear": "#bab0ac",
        "collaborative_refinement": "#59a14f",
        "conflict_modify": "#b279a2",
        "coupling_unclear": "#bab0ac",
    }
    for ax, dimension in zip(axes, ["outcome", "agency", "coupling"]):
        sub = label_counts[label_counts["dimension"] == dimension].set_index("label")
        ordered = [label for label in label_order[dimension] if label in sub.index]
        values = [100.0 * float(sub.loc[label, "share_of_modify"]) for label in ordered]
        bars = ax.bar(
            range(len(ordered)),
            values,
            color=[colors[label] for label in ordered],
            edgecolor="#333333",
            linewidth=0.6,
        )
        ax.set_title(dimension.capitalize())
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels([label_display[label] for label in ordered], rotation=25, ha="right")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.18)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    axes[0].set_ylabel("Share of modify trials (%)")
    fig.suptitle("Modification is heterogeneous", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "storyline2_modify_heterogeneity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    quadrant_order = ["AI_up_self_up", "AI_up_self_down", "AI_down_self_up", "AI_down_self_down"]
    quadrant_display = ["+/+", "+/-", "-/+", "-/-"]
    actions = ["accept_ai", "reject_ai", "modify"]
    action_display = ["accept", "reject", "modify"]
    counts = []
    nonzero_per_action = []
    for action in actions:
        grp = trials[trials["action"] == action]
        row = [(grp["quadrant"] == q).sum() for q in quadrant_order]
        counts.append(row)
        nonzero_per_action.append(sum(row))
    counts_arr = np.asarray(counts, dtype=float)
    shares = counts_arr / np.asarray(nonzero_per_action, dtype=float)[:, None] * 100.0

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    bottom = np.zeros(len(actions))
    quad_colors = ["#54a24b", "#f58518", "#4c78a8", "#e45756"]
    for idx, label in enumerate(quadrant_display):
        ax.bar(
            action_display,
            shares[:, idx],
            bottom=bottom,
            label=label,
            color=quad_colors[idx],
            edgecolor="#333333",
            linewidth=0.5,
        )
        bottom += shares[:, idx]
    for x, n in enumerate(nonzero_per_action):
        ax.text(x, 102, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Share among nonzero confidence updates (%)")
    ax.set_xlabel("")
    ax.set_title("Conjugate confidence quadrant distribution by behavior")
    ax.legend(title="Delta AI / Delta self", ncol=4, frameon=True, fontsize=8, title_fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "behavior_confidence_quadrant_distribution.png", dpi=240)
    plt.close(fig)

    metrics = ["modify_rate", "positive_modify_rate", "collaborative_modify_rate"]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    plot_data = participants.copy()
    median = plot_data["team_score"].median()
    plot_data["team_performer_group"] = np.where(plot_data["team_score"] >= median, "high", "low")
    means = plot_data.groupby("team_performer_group")[metrics].mean().reindex(["low", "high"])
    means.plot(kind="bar", ax=ax, color=["#f58518", "#e45756", "#72b7b2"])
    ax.set_ylabel("Participant-level rate")
    ax.set_xlabel("Team-score performer group")
    ax.set_title("Modify metrics by performer group")
    ax.set_ylim(0, max(0.35, float(means.max().max()) * 1.2))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "performer_modify_metrics.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    condition_colors = {1: "#4c78a8", 2: "#f58518"}
    for condition, grp in participants.groupby("condition"):
        ax.scatter(
            grp["positive_modify_rate"],
            grp["team_score"],
            s=35 + 120 * grp["accept_rate"],
            alpha=0.74,
            color=condition_colors.get(int(condition), "#777777"),
            edgecolor="#333333",
            linewidth=0.45,
            label=f"condition {int(condition)}",
        )
    sub = participants[["positive_modify_rate", "team_score"]].dropna()
    if len(sub) >= 2 and sub["positive_modify_rate"].nunique() > 1:
        slope, intercept, r_value, p_value, _ = stats.linregress(sub["positive_modify_rate"], sub["team_score"])
        x_line = np.linspace(sub["positive_modify_rate"].min(), sub["positive_modify_rate"].max(), 100)
        ax.plot(x_line, intercept + slope * x_line, color="#222222", linewidth=1.4)
        ax.text(
            0.02,
            0.97,
            f"r = {r_value:.2f}, p = {p_value:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
    ax.set_xlabel("Positive-outcome modify rate")
    ax.set_ylabel("Team score")
    ax.set_title("Positive modification and team performance")
    ax.grid(alpha=0.18)
    ax.legend(frameon=True, fontsize=8, title="Condition", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "team_score_positive_modify_scatter.png", dpi=240)
    plt.close(fig)

    coef_df = ols_table(participants, "team_score")
    terms = [
        "condition_2",
        "accept_rate",
        "modify_rate",
        "positive_modify_rate",
        "collaborative_modify_rate",
        "conflict_modify_rate",
    ]
    term_display = {
        "condition_2": "condition 2",
        "accept_rate": "accept rate",
        "modify_rate": "modify rate",
        "positive_modify_rate": "positive modify rate",
        "collaborative_modify_rate": "collaborative modify rate",
        "conflict_modify_rate": "conflict modify rate",
    }
    plot_df = coef_df[coef_df["term"].isin(terms)].copy()
    plot_df["lower"] = plot_df["coef"] - 1.96 * plot_df["std_error"]
    plot_df["upper"] = plot_df["coef"] + 1.96 * plot_df["std_error"]
    plot_df["term"] = pd.Categorical(plot_df["term"], categories=terms, ordered=True)
    plot_df = plot_df.sort_values("term")

    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    y = np.arange(len(plot_df))
    colors_coef = np.where(plot_df["coef"] >= 0, "#54a24b", "#e45756")
    ax.hlines(y, plot_df["lower"], plot_df["upper"], color="#555555", linewidth=1.4)
    ax.scatter(plot_df["coef"], y, color=colors_coef, s=46, zorder=3, edgecolor="#333333", linewidth=0.5)
    ax.axvline(0, color="#222222", linewidth=0.9)
    for yi, row in zip(y, plot_df.itertuples(index=False)):
        ax.text(
            row.coef,
            yi + 0.24,
            f"perm p={row.permutation_p:.3g}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([term_display[str(term)] for term in plot_df["term"]])
    ax.invert_yaxis()
    ax.set_xlabel("Regression coefficient for team score")
    ax.set_title("Team-score control model")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "team_score_regression_coefficients.png", dpi=240)
    plt.close(fig)


def write_notes() -> None:
    notes = """# Truss Modify Analysis

This folder analyzes modify trials in the complete truss detailed data.

Modify labels are deliberately organized as three paired dimensions rather than one mutually exclusive category:

- outcome: productive_modify vs unproductive_modify
- agency: self_correction vs partial_acceptance, with agency_unclear when neither definition fires
- coupling: collaborative_refinement vs conflict_modify, with coupling_unclear when neither definition fires

Operational definitions:

- productive_modify: modify trial with positive feedback.
- unproductive_modify: modify trial with negative feedback.
- self_correction: modify trial where self-confidence increases and AI-confidence does not increase.
- partial_acceptance: modify trial where AI-confidence increases and self-confidence does not decrease.
- collaborative_refinement: productive modify trial where both AI-confidence and self-confidence increase.
- conflict_modify: modify trial where AI-confidence and self-confidence move in opposite directions.

Confidence deltas use C[t+1] - C[t] for each trial.
"""
    (OUT_DIR / "README.md").write_text(notes, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials = build_trial_table()
    participants = participant_summary(trials)

    trials.to_csv(OUT_DIR / "trial_level_modify_analysis.csv", index=False)
    participants.to_csv(OUT_DIR / "participant_modify_summary.csv", index=False)
    summarize_action_feedback(trials).to_csv(OUT_DIR / "action_feedback_delta_summary.csv", index=False)
    summarize_confidence_deltas(trials).to_csv(OUT_DIR / "action_confidence_quadrants.csv", index=False)
    action_feedback_table(trials).to_csv(OUT_DIR / "table1_action_feedback.csv", index=False)
    action_confidence_quadrant_table(trials).to_csv(OUT_DIR / "table2_action_confidence_quadrants.csv", index=False)
    summarize_modify_labels(trials).to_csv(OUT_DIR / "modify_label_counts.csv", index=False)
    performer_summary(participants).to_csv(OUT_DIR / "performer_group_summary.csv", index=False)
    participant_correlations(participants).to_csv(OUT_DIR / "participant_correlations.csv", index=False)
    pd.concat(
        [ols_table(participants, "team_score"), ols_table(participants, "individual_score")],
        ignore_index=True,
    ).to_csv(OUT_DIR / "score_regression_controls.csv", index=False)
    save_plots(trials, participants)
    write_notes()

    print(f"Saved modify analysis outputs to {OUT_DIR}")
    print(f"Trials: {len(trials)}")
    print(f"Modify trials: {int((trials['action'] == 'modify').sum())}")


if __name__ == "__main__":
    main()

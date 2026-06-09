"""
Modify-case error analysis using the ORIGINAL 4-experience model (accept/reject).
Requires outputs/table1_model_params.csv from analyze_refined.py.
Outputs to outputs_original/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats

from original_common import (
    ConfigOrig,
    load_all_participants,
    build_analysis_matrices,
    load_model_params,
    compute_error_dataframe,
    bootstrap_ci,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_modify_errors(error_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    colors = {"accept": "#2ca02c", "reject": "#d62728", "modify": "#ff7f0e"}

    for action in ["accept", "reject", "modify"]:
        subset = error_df[error_df["action_label"] == action]
        ax1.hist(subset["ai_error_abs"], bins=30, alpha=0.6, label=action, color=colors.get(action, "#888"))
        ax2.hist(subset["self_error_abs"], bins=30, alpha=0.6, label=action, color=colors.get(action, "#888"))

    ax1.set_title("AI error |action")
    ax1.set_xlabel("Absolute error")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_title("Self error |action")
    ax2.set_xlabel("Absolute error")
    ax2.legend()
    ax2.grid(alpha=0.3)

    error_by_action = error_df.groupby("action_label")[["ai_error_abs", "self_error_abs"]].agg(["mean", "std", "count"])
    ai_means = error_by_action["ai_error_abs"]["mean"]
    ai_stds = error_by_action["ai_error_abs"]["std"]
    self_means = error_by_action["self_error_abs"]["mean"]
    self_stds = error_by_action["self_error_abs"]["std"]

    ax3.bar(ai_means.index, ai_means.values, yerr=ai_stds.values, capsize=5, color=[colors.get(a, "#888") for a in ai_means.index])
    ax3.set_title("AI mean abs error by action")
    ax3.grid(alpha=0.3, axis="y")

    ax4.bar(self_means.index, self_means.values, yerr=self_stds.values, capsize=5, color=[colors.get(a, "#888") for a in self_means.index])
    ax4.set_title("Self mean abs error by action")
    ax4.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "orig_modify_error_plots.png", dpi=ConfigOrig.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved orig_modify_error_plots.png")


def plot_modify_scatter_by_condition(error_df: pd.DataFrame, output_dir: Path) -> None:
    mod_df = error_df[error_df["action_label"] == "modify"].copy()
    if mod_df.empty:
        logger.info("No modify trials found; skip scatter plot.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row, cond in enumerate(sorted(mod_df["condition"].dropna().unique())):
        sub = mod_df[mod_df["condition"] == cond]
        axes[row, 0].scatter(sub["trial"], sub["ai_error_abs"], color="#1f77b4", alpha=0.7, s=30)
        axes[row, 0].axvline(20, color="orange", linestyle="--", linewidth=2, alpha=0.7)
        axes[row, 0].set_title(f"Cond {cond} - AI error (modify)")
        axes[row, 0].set_ylabel("Absolute error (normalized)")
        axes[row, 0].grid(alpha=0.3)

        axes[row, 1].scatter(sub["trial"], sub["self_error_abs"], color="#d62728", alpha=0.7, s=30)
        axes[row, 1].axvline(20, color="orange", linestyle="--", linewidth=2, alpha=0.7)
        axes[row, 1].set_title(f"Cond {cond} - Self error (modify)")
        axes[row, 1].grid(alpha=0.3)

    axes[1, 0].set_xlabel("Trial")
    axes[1, 1].set_xlabel("Trial")
    fig.suptitle("Modify trials: per-trial errors by condition (vertical line = perf change @20)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = output_dir / "orig_modify_error_scatter.png"
    fig.savefig(out_path, dpi=ConfigOrig.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_conf_trend_for_modify(trial_df: pd.DataFrame, output_dir: Path) -> None:
    subset = trial_df[trial_df["action_label"] == "modify"]
    if subset.empty:
        logger.info("No modify trials found; skip trend plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(axes, ["aiconf", "selfconf"], ["AI Confidence", "Self Confidence"]):
        mean = subset.groupby("trial")[col].mean()
        se = subset.groupby("trial")[col].std(ddof=1) / np.sqrt(subset.groupby("trial")[col].count())
        ax.errorbar(mean.index, mean.values, yerr=se.values, fmt="o-", color="black", ms=4, capsize=3, alpha=0.8)
        ax.axvline(20, color="orange", linestyle="--", linewidth=2, alpha=0.7, label="Perf change (trial 20)")
        ax.set_title(f"{title} (modify trials)")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out_path = output_dir / "orig_modify_conf_trend.png"
    fig.savefig(out_path, dpi=ConfigOrig.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def main() -> None:
    # Override output dir for this script
    output_dir = Path("original_modify_error")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data and params (from original model reproduction)
    participants, trial_df = load_all_participants(ConfigOrig.DATA_DIR)
    matrices = build_analysis_matrices(participants)
    model_params = load_model_params()

    error_df = compute_error_dataframe(matrices, trial_df, model_params)
    cond_map = trial_df[["pid", "condition"]].drop_duplicates()
    error_df = error_df.merge(cond_map, on="pid", how="left")

    modify_errors = error_df[error_df["action_label"] == "modify"]
    non_modify_errors = error_df[error_df["action_label"] != "modify"]

    ai_test = stats.mannwhitneyu(modify_errors["ai_error_abs"], non_modify_errors["ai_error_abs"], alternative="two-sided")
    self_test = stats.mannwhitneyu(modify_errors["self_error_abs"], non_modify_errors["self_error_abs"], alternative="two-sided")

    pid_modify = modify_errors.groupby("pid")[["ai_error_abs", "self_error_abs"]].mean()
    pid_non = non_modify_errors.groupby("pid")[["ai_error_abs", "self_error_abs"]].mean()
    pid_diff = pid_modify.join(pid_non, lsuffix="_modify", rsuffix="_nonmodify", how="inner")
    pid_diff["ai_diff"] = pid_diff["ai_error_abs_modify"] - pid_diff["ai_error_abs_nonmodify"]
    pid_diff["self_diff"] = pid_diff["self_error_abs_modify"] - pid_diff["self_error_abs_nonmodify"]
    ai_ci = bootstrap_ci(pid_diff["ai_diff"].dropna().values)
    self_ci = bootstrap_ci(pid_diff["self_diff"].dropna().values)

    pid_diff.to_csv(output_dir / "orig_modify_participant_level.csv", index=False)

    reg_cluster_rows = []
    for target in ["ai_error_abs", "self_error_abs"]:
        model = smf.ols(f"{target} ~ C(action_label)", data=error_df).fit(
            cov_type="cluster", cov_kwds={"groups": error_df["pid"]}
        )
        for term in ["Intercept", "C(action_label)[T.modify]", "C(action_label)[T.reject]"]:
            reg_cluster_rows.append({
                "target": target,
                "term": term,
                "coef": model.params.get(term, np.nan),
                "std_err": model.bse.get(term, np.nan),
                "p_value": model.pvalues.get(term, np.nan),
                "n": len(error_df),
            })
        logger.info(model.summary().as_text())
    pd.DataFrame(reg_cluster_rows).to_csv(output_dir / "orig_modify_cluster_regression.csv", index=False)

    followup = error_df[(error_df["prev_action"] == "modify") & error_df["prev_modify_magnitude"].notna()]
    follow_rows = []
    if not followup.empty:
        ai_model = smf.ols("ai_error_abs ~ prev_modify_magnitude", data=followup).fit(
            cov_type="cluster", cov_kwds={"groups": followup["pid"]}
        )
        self_model = smf.ols("self_error_abs ~ prev_modify_magnitude", data=followup).fit(
            cov_type="cluster", cov_kwds={"groups": followup["pid"]}
        )
        follow_rows = [
            {"target": "ai_error_abs", "coef": ai_model.params["prev_modify_magnitude"], "std_err": ai_model.bse["prev_modify_magnitude"], "p_value": ai_model.pvalues["prev_modify_magnitude"], "n": int(followup.shape[0])},
            {"target": "self_error_abs", "coef": self_model.params["prev_modify_magnitude"], "std_err": self_model.bse["prev_modify_magnitude"], "p_value": self_model.pvalues["prev_modify_magnitude"], "n": int(followup.shape[0])},
        ]
    pd.DataFrame(follow_rows).to_csv(output_dir / "orig_modify_followup_regression.csv", index=False)

    summary_lines = [
        "Original Model - Modification Error Analysis",
        "=" * 60,
        f"Modify trials: n={len(modify_errors)}, AI abs error mean={modify_errors['ai_error_abs'].mean():.4f}",
        f"Non-modify trials: n={len(non_modify_errors)}, AI abs error mean={non_modify_errors['ai_error_abs'].mean():.4f}",
        f"Mann-Whitney AI p={ai_test.pvalue:.4g}, Self p={self_test.pvalue:.4g}",
        f"Participant AI diff (modify-non): mean={pid_diff['ai_diff'].mean():.4f}, 95% CI [{ai_ci[0]:.4f}, {ai_ci[1]:.4f}], n={len(pid_diff)}",
        f"Participant Self diff (modify-non): mean={pid_diff['self_diff'].mean():.4f}, 95% CI [{self_ci[0]:.4f}, {self_ci[1]:.4f}], n={len(pid_diff)}",
    ]
    if follow_rows:
        summary_lines.append(f"Next-trial AI error ~ modify magnitude: coef={follow_rows[0]['coef']:.4f}, p={follow_rows[0]['p_value']:.4f}, n={follow_rows[0]['n']}")
    (output_dir / "orig_modify_error_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    plot_modify_errors(error_df, output_dir)
    plot_modify_scatter_by_condition(error_df, output_dir)
    plot_conf_trend_for_modify(trial_df, output_dir)


if __name__ == "__main__":
    main()

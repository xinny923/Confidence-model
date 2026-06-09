"""
Orig==AI alignment error analysis using the ORIGINAL 4-experience model.
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


def plot_align_errors(error_df: pd.DataFrame, output_dir: Path) -> None:
    groups = [
        ("orig==AI", error_df[error_df["orig_eq_ai"] == 1], "#4c72b0"),
        ("other", error_df[error_df["orig_eq_ai"] == 0], "#dd8452"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, target in zip(axes, ["ai_error_abs", "self_error_abs"]):
        for lbl, grp, color in groups:
            ax.hist(grp[target], bins=30, alpha=0.6, label=lbl, color=color)
        ax.set_title(f"{target} distribution")
        ax.set_xlabel("Absolute error")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "orig_align_error_hist.png", dpi=ConfigOrig.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved orig_align_error_hist.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [g[0] for g in groups]
    colors = [g[2] for g in groups]
    for ax, target in zip(axes, ["ai_error_abs", "self_error_abs"]):
        means = [g[1][target].mean() for g in groups]
        stds = [g[1][target].std() for g in groups]
        ax.bar(labels, means, yerr=stds, capsize=5, color=colors)
        ax.set_title(f"Mean {target}")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "orig_align_error_bars.png", dpi=ConfigOrig.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved orig_align_error_bars.png")


def plot_align_scatter_by_condition(error_df: pd.DataFrame, output_dir: Path) -> None:
    align_df = error_df[error_df["orig_eq_ai"] == 1].copy()
    if align_df.empty:
        logger.info("No orig==AI trials found; skip scatter plot.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row, cond in enumerate(sorted(align_df["condition"].dropna().unique())):
        sub = align_df[align_df["condition"] == cond]
        axes[row, 0].scatter(sub["trial"], sub["ai_error_abs"], color="#1f77b4", alpha=0.7, s=30)
        axes[row, 0].axvline(20, color="orange", linestyle="--", linewidth=2, alpha=0.7)
        axes[row, 0].set_title(f"Cond {cond} - AI error (orig==AI)")
        axes[row, 0].set_ylabel("Absolute error (normalized)")
        axes[row, 0].grid(alpha=0.3)

        axes[row, 1].scatter(sub["trial"], sub["self_error_abs"], color="#d62728", alpha=0.7, s=30)
        axes[row, 1].axvline(20, color="orange", linestyle="--", linewidth=2, alpha=0.7)
        axes[row, 1].set_title(f"Cond {cond} - Self error (orig==AI)")
        axes[row, 1].grid(alpha=0.3)

    axes[1, 0].set_xlabel("Trial")
    axes[1, 1].set_xlabel("Trial")
    fig.suptitle("Orig==AI trials: per-trial errors by condition (perf change @20)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = output_dir / "orig_align_error_scatter.png"
    fig.savefig(out_path, dpi=ConfigOrig.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_conf_trend_for_align(trial_df: pd.DataFrame, output_dir: Path) -> None:
    subset = trial_df[trial_df["orig_eq_ai"] == 1]
    if subset.empty:
        logger.info("No orig==AI trials found; skip trend plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(axes, ["aiconf", "selfconf"], ["AI Confidence", "Self Confidence"]):
        mean = subset.groupby("trial")[col].mean()
        se = subset.groupby("trial")[col].std(ddof=1) / np.sqrt(subset.groupby("trial")[col].count())
        ax.errorbar(mean.index, mean.values, yerr=se.values, fmt="o-", color="black", ms=4, capsize=3, alpha=0.8)
        ax.axvline(20, color="orange", linestyle="--", linewidth=2, alpha=0.7, label="Perf change (trial 20)")
        ax.set_title(f"{title} (orig==AI trials)")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out_path = output_dir / "orig_align_conf_trend.png"
    fig.savefig(out_path, dpi=ConfigOrig.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def main() -> None:
    output_dir = Path("original_align_error")
    output_dir.mkdir(parents=True, exist_ok=True)

    participants, trial_df = load_all_participants(ConfigOrig.DATA_DIR)
    matrices = build_analysis_matrices(participants)
    model_params = load_model_params()

    error_df = compute_error_dataframe(matrices, trial_df, model_params)
    cond_map = trial_df[["pid", "condition"]].drop_duplicates()
    error_df = error_df.merge(cond_map, on="pid", how="left")

    align_errors = error_df[error_df["orig_eq_ai"] == 1]
    non_align_errors = error_df[error_df["orig_eq_ai"] == 0]

    ai_test = stats.mannwhitneyu(align_errors["ai_error_abs"], non_align_errors["ai_error_abs"], alternative="two-sided")
    self_test = stats.mannwhitneyu(align_errors["self_error_abs"], non_align_errors["self_error_abs"], alternative="two-sided")

    pid_align = align_errors.groupby("pid")[["ai_error_abs", "self_error_abs"]].mean()
    pid_non = non_align_errors.groupby("pid")[["ai_error_abs", "self_error_abs"]].mean()
    pid_align_diff = pid_align.join(pid_non, lsuffix="_align", rsuffix="_nonalign", how="inner")
    pid_align_diff["ai_diff"] = pid_align_diff["ai_error_abs_align"] - pid_align_diff["ai_error_abs_nonalign"]
    pid_align_diff["self_diff"] = pid_align_diff["self_error_abs_align"] - pid_align_diff["self_error_abs_nonalign"]
    ai_ci = bootstrap_ci(pid_align_diff["ai_diff"].dropna().values)
    self_ci = bootstrap_ci(pid_align_diff["self_diff"].dropna().values)

    pid_align_diff.to_csv(output_dir / "orig_align_participant_level.csv", index=False)

    reg_rows = []
    for target in ["ai_error_abs", "self_error_abs"]:
        model = smf.ols(f"{target} ~ orig_eq_ai", data=error_df).fit(
            cov_type="cluster", cov_kwds={"groups": error_df["pid"]}
        )
        reg_rows.append({
            "target": target,
            "term": "orig_eq_ai",
            "coef": model.params.get("orig_eq_ai", np.nan),
            "std_err": model.bse.get("orig_eq_ai", np.nan),
            "p_value": model.pvalues.get("orig_eq_ai", np.nan),
            "n": len(error_df),
        })
        logger.info(model.summary().as_text())
    pd.DataFrame(reg_rows).to_csv(output_dir / "orig_align_regression.csv", index=False)

    followup = error_df[(error_df["prev_orig_eq_ai"] == 1) & error_df["prev_orig_eq_ai"].notna()]
    follow_rows = []
    if not followup.empty:
        ai_model = smf.ols("ai_error_abs ~ prev_orig_eq_ai", data=followup).fit(
            cov_type="cluster", cov_kwds={"groups": followup["pid"]}
        )
        self_model = smf.ols("self_error_abs ~ prev_orig_eq_ai", data=followup).fit(
            cov_type="cluster", cov_kwds={"groups": followup["pid"]}
        )
        follow_rows = [
            {"target": "ai_error_abs", "coef": ai_model.params["prev_orig_eq_ai"], "std_err": ai_model.bse["prev_orig_eq_ai"], "p_value": ai_model.pvalues["prev_orig_eq_ai"], "n": int(followup.shape[0])},
            {"target": "self_error_abs", "coef": self_model.params["prev_orig_eq_ai"], "std_err": self_model.bse["prev_orig_eq_ai"], "p_value": self_model.pvalues["prev_orig_eq_ai"], "n": int(followup.shape[0])},
        ]
    pd.DataFrame(follow_rows).to_csv(output_dir / "orig_align_followup_regression.csv", index=False)

    summary_lines = [
        "Original Model - Orig==AI Alignment Analysis",
        "=" * 60,
        f"Orig==AI trials: n={len(align_errors)}, AI abs error mean={align_errors['ai_error_abs'].mean():.4f}",
        f"Others: n={len(non_align_errors)}, AI abs error mean={non_align_errors['ai_error_abs'].mean():.4f}",
        f"Mann-Whitney AI p={ai_test.pvalue:.4g}, Self p={self_test.pvalue:.4g}",
        f"Participant AI diff (align-other): mean={pid_align_diff['ai_diff'].mean():.4f}, 95% CI [{ai_ci[0]:.4f}, {ai_ci[1]:.4f}], n={len(pid_align_diff)}",
        f"Participant Self diff (align-other): mean={pid_align_diff['self_diff'].mean():.4f}, 95% CI [{self_ci[0]:.4f}, {self_ci[1]:.4f}], n={len(pid_align_diff)}",
    ]
    if follow_rows:
        summary_lines.append(f"Next-trial AI error ~ I(orig==AI)(n): coef={follow_rows[0]['coef']:.4f}, p={follow_rows[0]['p_value']:.4f}, n={follow_rows[0]['n']}")
    (output_dir / "orig_align_error_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    plot_align_errors(error_df, output_dir)
    plot_align_scatter_by_condition(error_df, output_dir)
    plot_conf_trend_for_align(trial_df, output_dir)


if __name__ == "__main__":
    main()

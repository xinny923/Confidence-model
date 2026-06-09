"""
Plot AI/self confidence trajectories by condition with performance change marker.
Mimics Fig.4 style: mean over participants, vertical line at trial 20.
Uses original 2-action model data (Data/), outputs to output_benchmark/.
"""

from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from original_common import ConfigOrig, load_all_participants

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    out_dir = Path("output_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    participants, _ = load_all_participants(ConfigOrig.DATA_DIR)

    # Stack confidence by condition
    ai_conf = np.vstack([p.c_series for p in participants])  # shape (100, 31)
    self_conf = np.vstack([p.self_series for p in participants])
    cond = np.array([p.condition for p in participants])

    trials = np.arange(ConfigOrig.NUM_TRIALS + 1)  # include practice idx 0
    change_trial = 20  # performance switch as in paper

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for row, condition in enumerate([1, 2]):
        mask = cond == condition
        ai_mean = np.nanmean(ai_conf[mask], axis=0)
        ai_se = np.nanstd(ai_conf[mask], axis=0, ddof=1) / np.sqrt(mask.sum())
        self_mean = np.nanmean(self_conf[mask], axis=0)
        self_se = np.nanstd(self_conf[mask], axis=0, ddof=1) / np.sqrt(mask.sum())

        ax_ai = axes[row, 0]
        ax_self = axes[row, 1]

        ax_ai.errorbar(trials, ai_mean, yerr=ai_se, fmt="o", ms=3, color="black", alpha=0.6, capsize=2, label="AI conf")
        ax_ai.axvline(change_trial, color="orange", linestyle="--", linewidth=2, alpha=0.7, label="Perf change" if row == 0 else None)
        ax_ai.set_title(f"Condition {condition} - AI Confidence")
        ax_ai.set_ylabel("Confidence")
        ax_ai.grid(alpha=0.3)
        ax_ai.set_ylim(0, 1)

        ax_self.errorbar(trials, self_mean, yerr=self_se, fmt="o", ms=3, color="black", alpha=0.6, capsize=2, label="Self conf")
        ax_self.axvline(change_trial, color="orange", linestyle="--", linewidth=2, alpha=0.7, label="Perf change" if row == 0 else None)
        ax_self.set_title(f"Condition {condition} - Self Confidence")
        ax_self.grid(alpha=0.3)
        ax_self.set_ylim(0, 1)

    axes[1, 0].set_xlabel("Trial")
    axes[1, 1].set_xlabel("Trial")
    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    fig.suptitle("AI/Self Confidence Trajectories with Performance Change (Trial 20)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    out_path = out_dir / "ai_self_conf_trends.png"
    fig.savefig(out_path, dpi=ConfigOrig.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()

"""
Analyze whether accumulated reward evidence predicts Modify/Reject actions.

For each participant and trial:
  - action_label: accept / reject / modify (final vs ai vs orig)
  - feedback2 normalized: +1 / -1 (NaN ignored)
  - cumulative_reward: sum of normalized feedback up to previous trial (trial 1 uses 0)

Outputs (in output_behavior/):
  - action_probs_by_bin.csv : probability of modify/reject by cumulative_reward bins
  - logit_coeffs.csv        : logistic regression coefficients
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Config:
    NUM_TRIALS = 30
    PRACTICE_LAST_IDX = 4
    MAIN_START_IDX = 6
    SCORE_ROW_IDX = MAIN_START_IDX + NUM_TRIALS

    DATA_DIR = Path("Data")
    OUTPUT_DIR = Path("output_behavior")


def safe_float(value: Any, default: float = math.nan) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"none", "nan"}:
            return default
        try:
            return float(text)
        except ValueError:
            return default
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        return float(value)
    except Exception:
        return default


def safe_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def parse_filename(path: Path) -> Tuple[int, int]:
    import re

    m = re.search(r"data(\d+)_(\d+)\.csv", path.name)
    if not m:
        raise ValueError(f"Bad filename {path.name}")
    return int(m.group(1)), int(m.group(2))


def load_trials() -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for path in sorted(Config.DATA_DIR.glob("data*_*.csv")):
        base_id, condition = parse_filename(path)
        df = pd.read_csv(path, header=None)
        for t in range(Config.NUM_TRIALS):
            r = Config.MAIN_START_IDX + t
            orig = safe_str(df.iat[r, 2])
            ai = safe_str(df.iat[r, 3])
            final = safe_str(df.iat[r, 5])
            fb = safe_float(df.iat[r, 7])
            action_label = "accept" if final == ai else ("reject" if final == orig else "modify")
            records.append(
                {
                    "base_id": base_id,
                    "condition": condition,
                    "trial": t + 1,
                    "orig": orig,
                    "ai": ai,
                    "final": final,
                    "action_label": action_label,
                    "feedback_raw": fb,
                    "feedback_norm": (fb / 5.0) if not math.isnan(fb) else math.nan,
                }
            )
    return pd.DataFrame(records)


def add_cumulative_reward(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cum_reward"] = 0.0
    df["cum_diff_ai_minus_user"] = 0.0
    df["cum_ai_reward"] = 0.0
    df["cum_user_reward"] = 0.0
    for bid, sub in df.groupby("base_id"):
        sub = sub.sort_values("trial")
        cum = []
        cum_diff = []
        cum_ai = []
        cum_user = []
        total = 0.0
        ai_accum = 0.0
        user_accum = 0.0
        for _, row in sub.iterrows():
            fb = row["feedback_norm"]
            # cumulative up to previous trial
            cum.append(total)
            cum_diff.append(ai_accum - user_accum)
            cum_ai.append(ai_accum)
            cum_user.append(user_accum)
            if not math.isnan(fb):
                if row["final"] == row["ai"]:
                    ai_accum += fb
                else:
                    user_accum += fb
                total += fb
        df.loc[sub.index, "cum_reward"] = cum
        df.loc[sub.index, "cum_diff_ai_minus_user"] = cum_diff
        df.loc[sub.index, "cum_ai_reward"] = cum_ai
        df.loc[sub.index, "cum_user_reward"] = cum_user
    return df


def bin_probabilities(df: pd.DataFrame, col: str = "cum_reward", n_bins: int = 6) -> pd.DataFrame:
    df = df.copy()
    df["bin"] = pd.qcut(df[col], n_bins, duplicates="drop")
    rows = []
    for b, sub in df.groupby("bin"):
        n = len(sub)
        p_mod = (sub["action_label"] == "modify").mean()
        p_rej = (sub["action_label"] == "reject").mean()
        rows.append({"bin": str(b), "count": n, "p_modify": p_mod, "p_reject": p_rej,
                     "bin_min": sub[col].min(), "bin_max": sub[col].max(),
                     "bin_mean": sub[col].mean()})
    return pd.DataFrame(rows)


def run_logit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_modify"] = (df["action_label"] == "modify").astype(int)
    df["is_reject"] = (df["action_label"] == "reject").astype(int)
    res_list = []
    for target in ["is_modify", "is_reject"]:
        model = smf.logit(f"{target} ~ cum_reward", data=df)
        try:
            fit = model.fit(disp=False)
            table = fit.summary2().tables[1].reset_index().rename(columns={"index": "term"})
            table.insert(0, "target", target)
            res_list.append(table[["target", "term", "Coef.", "Std.Err.", "z", "P>|z|"]])
        except Exception as e:
            logger.warning("Logit failed for %s: %s", target, e)
    if res_list:
        return pd.concat(res_list, ignore_index=True)
    return pd.DataFrame()

def plot_dual_per_condition(trials: pd.DataFrame, cond: int, target: str, bins: int, output_path: Path):
    """
    For a single condition and target (modify/reject):
      - bottom x: cumulative AI reward (binned), line = rate vs AI reward
      - top x: cumulative human reward (binned), line = rate vs human reward
    """
    import matplotlib.pyplot as plt

    sub = trials[trials["condition"] == cond].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # AI bins
    sub["bin_ai"] = pd.qcut(sub["cum_ai_reward"], bins, duplicates="drop")
    grp_ai = sub.groupby("bin_ai").agg(
        rate=("action_label", lambda x: (x == target).mean()),
        x_mean=("cum_ai_reward", "mean"),
    )
    ax.plot(grp_ai["x_mean"], grp_ai["rate"], marker="o", color="#1f77b4", label="AI reward")
    ax.set_xlabel("Cumulative AI reward (up to previous trial)")
    ax.set_ylabel(f"{target.capitalize()} rate")
    ax.grid(alpha=0.3)
    ax.set_title(f"Cond {cond}: {target.capitalize()} vs cumulative rewards")

    # Human bins on top axis
    ax_top = ax.twiny()
    sub["bin_h"] = pd.qcut(sub["cum_user_reward"], bins, duplicates="drop")
    grp_h = sub.groupby("bin_h").agg(
        rate=("action_label", lambda x: (x == target).mean()),
        x_mean=("cum_user_reward", "mean"),
    )
    ax_top.plot(grp_h["x_mean"], grp_h["rate"], marker="s", color="#d62728", label="Human reward")
    ax_top.set_xlabel("Cumulative human reward (up to previous trial)")
    ax_top.set_xlim(grp_h["x_mean"].min(), grp_h["x_mean"].max())

    # Combine legends
    lines = ax.get_lines() + ax_top.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trials = load_trials()
    trials = add_cumulative_reward(trials)

    # Save bin probabilities
    probs = bin_probabilities(trials)
    probs.to_csv(Config.OUTPUT_DIR / "action_probs_by_bin.csv", index=False)
    logger.info("Saved bin probabilities: %s", Config.OUTPUT_DIR / "action_probs_by_bin.csv")

    # Bin probabilities for cumulative diff (AI - user)
    probs_diff = bin_probabilities(trials, col="cum_diff_ai_minus_user")
    probs_diff.to_csv(Config.OUTPUT_DIR / "action_probs_by_diff.csv", index=False)
    logger.info("Saved bin probabilities (AI-user diff): %s", Config.OUTPUT_DIR / "action_probs_by_diff.csv")

    # Logistic regression
    logit = run_logit(trials)
    if not logit.empty:
        logit.to_csv(Config.OUTPUT_DIR / "logit_coeffs.csv", index=False)
        logger.info("Saved logit coefficients: %s", Config.OUTPUT_DIR / "logit_coeffs.csv")

    # Plot: modification rate vs cumulative reward (binned), by condition
    try:
        import matplotlib.pyplot as plt

        bins = 20
        trials["bin"] = pd.qcut(trials["cum_reward"], bins, duplicates="drop")
        # Per-condition dual-axis plots (AI vs Human cumulative rewards), modify/reject
        for cond in sorted(trials["condition"].unique()):
            out_m = Config.OUTPUT_DIR / f"modify_rate_cond{cond}_dual.png"
            out_r = Config.OUTPUT_DIR / f"reject_rate_cond{cond}_dual.png"
            plot_dual_per_condition(trials, cond=cond, target="modify", bins=bins, output_path=out_m)
            plot_dual_per_condition(trials, cond=cond, target="reject", bins=bins, output_path=out_r)
            logger.info("Saved plot: %s", out_m)
            logger.info("Saved plot: %s", out_r)

    except Exception as e:
        logger.warning("Plotting failed: %s", e)

    # Quick counts for reference
    counts = trials["action_label"].value_counts(normalize=True)
    logger.info("Action distribution: %s", counts.to_dict())
    logger.info("Cumulative reward range: %.3f to %.3f", trials["cum_reward"].min(), trials["cum_reward"].max())


if __name__ == "__main__":
    main()

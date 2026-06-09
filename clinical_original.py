"""
Clinical diagnostics for the original 4-term model (mdiscrete_core).
Reads parameters from output_benchmark/extended_model_params.csv (Chong-style fit),
runs forward, and reports observed vs predicted plus residuals on selected trials
with action distributions. No velocity state in the original model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent
MOM_PATH = ROOT / "momentum"
if str(MOM_PATH) not in sys.path:
    sys.path.append(str(MOM_PATH))
import mdiscrete_core as core

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PARAM_PATH = Path("output_benchmark") / "table1_model_params.csv"
OUTPUT_PATH = Path("output_benchmark") / "clinical_report_original.txt"

# Trial sets (1-based)
TRIALS = {
    "cond1_ai": [3, 5, 7, 13, 14, 16, 20, 21, 24, 26],
    "cond1_self": [5, 7, 19, 26],
    "cond2_ai": [1, 2, 3, 6, 7, 8, 13, 14, 19, 24],
    "cond2_self": [2, 5, 6, 7, 15, 22, 24],
}


def load_params(path: Path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path)
    params = {}
    cols = [c for c in core.Config.PARAMETER_NAMES if c in df.columns]
    for name in ["aiconf", "selfconf"]:
        params[name] = df[df["model"] == name].iloc[0][cols].values
    return params


def simulate_original(params: np.ndarray, observed: np.ndarray, e_tensor: np.ndarray) -> np.ndarray:
    """
    Run the original 4-term model (no velocity, no align) to get predictions only.
    """
    alpha_e, alpha_a, alpha_b, o1, o2, o3, o4, gamma = params
    n_participants, series_len = observed.shape
    n_trials = series_len - 1

    c = observed[:, 0].copy()
    b = observed[:, 0].copy()
    a = observed[:, 0].copy()
    pred = np.zeros_like(observed)
    pred[:, 0] = c

    for t in range(n_trials):
        if t > 0:
            a = gamma * pred[:, t] + (1 - gamma) * a
        experience = (
            o1 * e_tensor[:, t, 0] +
            o2 * e_tensor[:, t, 1] +
            o3 * e_tensor[:, t, 2] +
            o4 * e_tensor[:, t, 3]
        )
        c = c + alpha_e * (experience - c) + alpha_a * (a - c) + alpha_b * (b - c)
        pred[:, t + 1] = c
    return pred


def summarize_trials(label: str, trials: List[int], cond: int, obs: np.ndarray, pred: np.ndarray,
                     condition_array: np.ndarray, trial_df: pd.DataFrame) -> List[str]:
    idx = np.where(condition_array == cond)[0]
    lines = []
    lines.append(f"\n## {label} (Cond{cond})")
    lines.append("trial | obs_mean | pred_mean | resid | actions")
    lines.append("-" * 80)
    for t in trials:
        obs_mean = obs[idx, t].mean()
        pred_mean = pred[idx, t].mean()
        resid = pred_mean - obs_mean
        actions = trial_df[(trial_df["condition"] == cond) & (trial_df["trial"] == t)]["action_label"]
        action_counts = actions.value_counts().to_dict()
        action_str = ", ".join([f"{k}:{v}" for k, v in action_counts.items()])
        lines.append(f"{t:5d} | {obs_mean:8.3f} | {pred_mean:9.3f} | {resid:+6.3f} | {action_str}")
    return lines


def main():
    params = load_params(PARAM_PATH)
    parts, trial_df = core.load_all_participants(core.Config.DATA_DIR)
    matrices = core.build_analysis_matrices(parts)

    pred_ai = simulate_original(params["aiconf"], matrices["ai_conf_matrix"], matrices["e_tensor"])
    pred_self = simulate_original(params["selfconf"], matrices["self_conf_matrix"], matrices["e_tensor"])

    report: List[str] = []
    report.append("Clinical diagnostics - original 4-term model")
    report.append(f"Params loaded from: {PARAM_PATH}")

    report += summarize_trials("AI confidence", TRIALS["cond1_ai"], 1, matrices["ai_conf_matrix"], pred_ai, matrices["condition_array"], trial_df)
    report += summarize_trials("Self confidence", TRIALS["cond1_self"], 1, matrices["self_conf_matrix"], pred_self, matrices["condition_array"], trial_df)
    report += summarize_trials("AI confidence", TRIALS["cond2_ai"], 2, matrices["ai_conf_matrix"], pred_ai, matrices["condition_array"], trial_df)
    report += summarize_trials("Self confidence", TRIALS["cond2_self"], 2, matrices["self_conf_matrix"], pred_self, matrices["condition_array"], trial_df)

    OUTPUT_PATH.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nReport saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

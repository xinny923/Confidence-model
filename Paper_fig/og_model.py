"""
og_model: clean standalone version of the original MATLAB benchmark.

This is the 8-parameter, 4-experience-channel model used by Code/myfun.m:
  x = [alpha_e, alpha_a, alpha_b, omega1, omega2, omega3, omega4, gamma]
  E = omega1*accept+ + omega2*reject+ + omega3*accept- + omega4*reject-

Outputs (default):
  - figure_conf.png
  - og_model_params.csv
Optional: robustness + VIF when enabled.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math
import re
import difflib

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from metric_utils import condition_mean_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Config:
    NUM_TRIALS = 30
    PRACTICE_LAST_IDX = 4
    MAIN_START_IDX = 6
    SCORE_ROW_IDX = MAIN_START_IDX + NUM_TRIALS
    PERF_CHANGE_TRIAL = 20

    PARAMETER_NAMES = [
        "alpha_e", "alpha_a", "alpha_b",
        "omega1", "omega2", "omega3", "omega4",
        "gamma",
    ]
    INITIAL_GUESS = {
        "aiconf": [0.267181677384535, 0.340450953120926, 0.0524238420736158, 0.843879167378967, 0.211531837817135, 0.0, 0.521737124180336, 0.389670818141326],
        "selfconf": [0.284419387583965, 0.470608119908973, 0.0, 0.573623483940813, 0.828368714233239, 0.238391875829198, 0.286322496033589, 0.114727712177294],
    }
    MODIFY_AI = 0.0
    MODIFY_SELF = 1.0

    MAX_NFEV = 5000
    RANDOM_SEED = 42
    ROBUSTNESS_N_ITERATIONS = 100
    ROBUSTNESS_SUBSET_SIZE = 80
    ROBUSTNESS_ENABLED = True
    VIF_ANALYSIS = True

    DATA_DIR = Path("Data")
    OUTPUT_DIR = Path("Paper_fig") / "og_output"
    PLOT_DPI = 300


def safe_float(value: Any, default: float = math.nan) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "nan"}:
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
    match = re.search(r"data(\d+)_(\d+)\.csv", path.name)
    if not match:
        raise ValueError(f"Invalid filename format: {path.name}")
    return int(match.group(1)), int(match.group(2))


def compute_modification_magnitude(original_move: str, ai_suggestion: str, final_move: str) -> float:
    if final_move == ai_suggestion or final_move == original_move:
        return 0.0
    sim_ai = difflib.SequenceMatcher(None, str(final_move), str(ai_suggestion)).ratio() if ai_suggestion else 0.0
    sim_orig = difflib.SequenceMatcher(None, str(final_move), str(original_move)).ratio() if original_move else 0.0
    return 1.0 - max(sim_ai, sim_orig)


@dataclass
class ParticipantData:
    pid: int
    base_id: int
    condition: int
    c_series: np.ndarray
    self_series: np.ndarray
    e_matrix: np.ndarray
    act_series: np.ndarray
    perf_series: np.ndarray
    skill_score: float
    team_score: float
    trial_records: List[Dict[str, Any]]


def parse_participant_data(path: Path, pid: int, condition: int, base_id: int) -> ParticipantData:
    df = pd.read_csv(path, header=None)

    def get_num(row: int, col: int) -> float:
        return safe_float(df.iat[row, col])

    def get_str(row: int, col: int) -> str:
        return safe_str(df.iat[row, col])

    c_series = np.array(
        [get_num(Config.PRACTICE_LAST_IDX, 9)] +
        [get_num(Config.MAIN_START_IDX + t, 9) for t in range(Config.NUM_TRIALS)],
        dtype=float
    )
    self_series = np.array(
        [get_num(Config.PRACTICE_LAST_IDX, 8)] +
        [get_num(Config.MAIN_START_IDX + t, 8) for t in range(Config.NUM_TRIALS)],
        dtype=float
    )

    e_matrix = np.zeros((Config.NUM_TRIALS, 4), dtype=float)
    act_values = [0]
    trial_records: List[Dict[str, Any]] = []

    for t in range(Config.NUM_TRIALS):
        row_idx = Config.MAIN_START_IDX + t
        original_move = get_str(row_idx, 2)
        ai_suggestion = get_str(row_idx, 3)
        final_move = get_str(row_idx, 5)
        feedback2 = get_num(row_idx, 7)

        if final_move == ai_suggestion:
            action = 0
        elif final_move == original_move:
            action = 1
        else:
            action = 2

        modify_magnitude = compute_modification_magnitude(original_move, ai_suggestion, final_move)
        orig_eq_ai = int(original_move == ai_suggestion)

        act_values.append(action)

        if not math.isnan(feedback2):
            sign = int(feedback2 / 5 * -1)
            if sign == -1:
                if action == 0:
                    e_matrix[t, 0] = 1.0
                elif action == 1:
                    e_matrix[t, 1] = 1.0
                else:
                    e_matrix[t, 0] = Config.MODIFY_AI
                    e_matrix[t, 1] = Config.MODIFY_SELF
            else:
                if action == 0:
                    e_matrix[t, 2] = 1.0
                elif action == 1:
                    e_matrix[t, 3] = 1.0
                else:
                    e_matrix[t, 2] = Config.MODIFY_AI
                    e_matrix[t, 3] = Config.MODIFY_SELF

        trial_records.append({
            "pid": pid,
            "condition": condition,
            "trial": t + 1,
            "original_move": original_move,
            "ai_suggestion": ai_suggestion,
            "final_move": final_move,
            "action": action,
            "action_label": ["accept", "reject", "modify"][action],
            "modify_magnitude": modify_magnitude,
            "orig_eq_ai": orig_eq_ai,
            "feedback1": get_num(row_idx, 6),
            "feedback2": feedback2,
            "selfconf": get_num(row_idx, 8),
            "aiconf": get_num(row_idx, 9),
        })

    perf_series = np.array(
        [get_num(Config.PRACTICE_LAST_IDX, 7)] +
        [get_num(Config.MAIN_START_IDX + t, 7) for t in range(Config.NUM_TRIALS)],
        dtype=float
    )
    skill_score = np.nansum([get_num(Config.MAIN_START_IDX + t, 6) for t in range(Config.NUM_TRIALS)])
    team_score = get_num(Config.SCORE_ROW_IDX, 1)

    return ParticipantData(
        pid=pid,
        base_id=base_id,
        condition=condition,
        c_series=c_series,
        self_series=self_series,
        e_matrix=e_matrix,
        act_series=np.array(act_values, dtype=float),
        perf_series=perf_series,
        skill_score=skill_score,
        team_score=team_score,
        trial_records=trial_records,
    )


def load_all_participants(data_dir: Path) -> Tuple[List[ParticipantData], pd.DataFrame]:
    file_mapping: Dict[int, Dict[int, Path]] = {1: {}, 2: {}}
    for path in sorted(data_dir.glob("data*_*.csv")):
        base_id, condition = parse_filename(path)
        file_mapping[condition][base_id] = path

    participants: List[ParticipantData] = []
    all_trial_records: List[Dict[str, Any]] = []
    pid_counter = 1

    for condition in (1, 2):
        for base_id in sorted(file_mapping[condition]):
            participant = parse_participant_data(file_mapping[condition][base_id], pid_counter, condition, base_id)
            participants.append(participant)
            all_trial_records.extend(participant.trial_records)
            pid_counter += 1

    return participants, pd.DataFrame(all_trial_records)


def build_analysis_matrices(participants: List[ParticipantData]) -> Dict[str, np.ndarray]:
    return {
        "ai_conf_matrix": np.vstack([p.c_series for p in participants]),
        "self_conf_matrix": np.vstack([p.self_series for p in participants]),
        "e_tensor": np.stack([p.e_matrix for p in participants]),
        "pid_array": np.array([p.pid for p in participants], dtype=int),
        "condition_array": np.array([p.condition for p in participants], dtype=int),
    }


def simulate_confidence_dynamics_extended(params: np.ndarray, observed: np.ndarray, e_tensor: np.ndarray) -> np.ndarray:
    alpha_e, alpha_a, alpha_b, omega1, omega2, omega3, omega4, gamma = params
    n_participants, series_len = observed.shape
    n_trials = series_len - 1

    c = observed[:, 0].copy()
    b = observed[:, 0].copy()
    a = observed[:, 0].copy()
    predicted = np.zeros_like(observed)
    predicted[:, 0] = c

    for t in range(n_trials):
        if t > 0:
            # MATLAB Code/myfun.m uses C(:, i-1) inside a 1-indexed loop after
            # C has already been initialized with C0. Thus trial 2 still uses
            # C0, trial 3 uses prediction after trial 1, and so on.
            a = gamma * predicted[:, t - 1] + (1 - gamma) * a
        experience = (
            omega1 * e_tensor[:, t, 0] +
            omega2 * e_tensor[:, t, 1] +
            omega3 * e_tensor[:, t, 2] +
            omega4 * e_tensor[:, t, 3]
        )
        c = c + alpha_e * (experience - c) + alpha_a * (a - c) + alpha_b * (b - c)
        predicted[:, t + 1] = c

    residuals = observed[:, 1:] - predicted[:, 1:]
    return residuals.T.ravel()


def compute_model_predictions_extended(params: np.ndarray, observed: np.ndarray, e_tensor: np.ndarray) -> np.ndarray:
    alpha_e, alpha_a, alpha_b, omega1, omega2, omega3, omega4, gamma = params
    n_participants, series_len = observed.shape
    n_trials = series_len - 1

    c = observed[:, 0].copy()
    b = observed[:, 0].copy()
    a = observed[:, 0].copy()
    predicted = np.zeros_like(observed)
    predicted[:, 0] = c

    for t in range(n_trials):
        if t > 0:
            a = gamma * predicted[:, t - 1] + (1 - gamma) * a
        experience = (
            omega1 * e_tensor[:, t, 0] +
            omega2 * e_tensor[:, t, 1] +
            omega3 * e_tensor[:, t, 2] +
            omega4 * e_tensor[:, t, 3]
        )
        c = c + alpha_e * (experience - c) + alpha_a * (a - c) + alpha_b * (b - c)
        predicted[:, t + 1] = c

    return predicted


def fit_extended_model(
    observed: np.ndarray,
    e_tensor: np.ndarray,
    initial_params: np.ndarray,
    condition_array: np.ndarray,
) -> Dict[str, Any]:
    bounds = (np.zeros_like(initial_params), np.ones_like(initial_params))
    result = least_squares(
        simulate_confidence_dynamics_extended,
        x0=initial_params,
        bounds=bounds,
        args=(observed, e_tensor),
        max_nfev=Config.MAX_NFEV,
    )

    pred_all = compute_model_predictions_extended(result.x, observed, e_tensor)
    metrics = condition_mean_metrics(
        observed, pred_all, condition_array, Config.NUM_TRIALS, adjustment_k=9
    )

    return {"params": result.x, "mse": metrics["mse"], "r2": metrics["r2"], "adj_r2": metrics["adj_r2"]}


def fit_all_models(matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
    records = []
    for model_name, conf_matrix in [
        ("aiconf", matrices["ai_conf_matrix"]),
        ("selfconf", matrices["self_conf_matrix"]),
    ]:
        initial = np.array(Config.INITIAL_GUESS[model_name], dtype=float)
        results = fit_extended_model(conf_matrix, matrices["e_tensor"], initial, matrices["condition_array"])
        record = {
            "model": model_name,
            "mse": results["mse"],
            "r2": results["r2"],
            "adj_r2": results["adj_r2"],
        }
        record.update(dict(zip(Config.PARAMETER_NAMES, results["params"])))
        records.append(record)
    return pd.DataFrame(records)


def plot_conf(pred_ai: np.ndarray, pred_self: np.ndarray,
              data_ai: np.ndarray, data_self: np.ndarray,
              condition_array: np.ndarray) -> None:
    trials = np.arange(0, Config.NUM_TRIALS + 1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    titles = {(0, 0): "A", (0, 1): "B", (1, 0): "C", (1, 1): "D"}
    change_point = Config.PERF_CHANGE_TRIAL

    for col, cond in enumerate([1, 2]):
        idx = np.where(condition_array == cond)[0]
        if len(idx) == 0:
            continue

        if cond == 1:
            left_label = "High-performing AI\n(80% accuracy)"
            right_label = "Low-performing AI\n(20% accuracy)"
            left_color = "#cfe1f2"
            right_color = "#f7e6c6"
        else:
            left_label = "Low-performing AI\n(20% accuracy)"
            right_label = "High-performing AI\n(80% accuracy)"
            left_color = "#f7e6c6"
            right_color = "#cfe1f2"

        for row, (obs_mat, pred_mat, ylab) in enumerate([
            (data_ai, pred_ai, "Confidence in AI"),
            (data_self, pred_self, "Self-confidence"),
        ]):
            obs = obs_mat[idx]
            pred = pred_mat[idx]
            mean_obs = np.nanmean(obs, axis=0)
            se_obs = np.nanstd(obs, axis=0, ddof=1) / np.sqrt(obs.shape[0])
            mean_pred = np.nanmean(pred, axis=0)
            ax = axes[row, col]

            ax.axvspan(0, change_point, ymin=0.0, ymax=0.12, color=left_color, alpha=0.7, zorder=0)
            ax.axvspan(change_point, Config.NUM_TRIALS, ymin=0.0, ymax=0.12, color=right_color, alpha=0.7, zorder=0)
            ax.text((0 + change_point) / 2, 0.06, left_label, ha="center", va="center", fontsize=9)
            ax.text((change_point + Config.NUM_TRIALS) / 2, 0.06, right_label, ha="center", va="center", fontsize=9)

            ax.axvline(change_point, color="#ff7f0e", linewidth=1.2, label="AI performance change")
            ax.errorbar(trials, mean_obs, yerr=se_obs, fmt="o", color="black",
                        markersize=4, capsize=2, label="Data")
            ax.plot(trials, mean_pred, color="#1f77b4", linewidth=2.2, label="Average model fit")

            ax.set_ylim(0, 1)
            ax.set_xlim(0, Config.NUM_TRIALS)
            ax.grid(alpha=0.3)
            ax.set_title(titles[(row, col)], loc="left", fontweight="bold")
            if row == 1:
                ax.set_xlabel("Puzzle number, n")
            ax.set_ylabel(ylab)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", frameon=True, framealpha=1.0, fontsize=9)

    fig.tight_layout()
    fig.savefig(Config.OUTPUT_DIR / "figure_conf.png", dpi=Config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def run_robustness(matrices: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n_iter = Config.ROBUSTNESS_N_ITERATIONS
    subset_size = Config.ROBUSTNESS_SUBSET_SIZE

    rng = np.random.default_rng(Config.RANDOM_SEED)
    n_participants = matrices["ai_conf_matrix"].shape[0]
    subset_size = min(subset_size, n_participants)

    results = {
        "aiconf": {"params": [], "mse": [], "r2": [], "adj_r2": []},
        "selfconf": {"params": [], "mse": [], "r2": [], "adj_r2": []},
    }

    for i in range(n_iter):
        if (i + 1) % max(1, n_iter // 10) == 0:
            logger.info("Robustness progress: %s/%s", i + 1, n_iter)
        idx = rng.choice(n_participants, size=subset_size, replace=False)
        idx.sort()
        e_sub = matrices["e_tensor"][idx]
        cond_sub = matrices["condition_array"][idx]

        for model_name, conf_matrix in [
            ("aiconf", matrices["ai_conf_matrix"]),
            ("selfconf", matrices["self_conf_matrix"]),
        ]:
            obs_sub = conf_matrix[idx]
            initial = np.array(Config.INITIAL_GUESS[model_name], dtype=float)
            fit_res = fit_extended_model(obs_sub, e_sub, initial, cond_sub)
            results[model_name]["params"].append(fit_res["params"])
            results[model_name]["mse"].append(fit_res["mse"])
            results[model_name]["r2"].append(fit_res["r2"])
            results[model_name]["adj_r2"].append(fit_res["adj_r2"])

    for model_name in results:
        results[model_name]["params"] = np.array(results[model_name]["params"])
        results[model_name]["mse"] = np.array(results[model_name]["mse"])
        results[model_name]["r2"] = np.array(results[model_name]["r2"])
        results[model_name]["adj_r2"] = np.array(results[model_name]["adj_r2"])
    return results


def compute_vif(params_df: pd.DataFrame) -> pd.DataFrame:
    cols = params_df.columns.tolist()
    x = params_df.values.astype(float)
    vif_records = []
    for i, name in enumerate(cols):
        y = x[:, i]
        x_others = np.delete(x, i, axis=1)
        x_others = np.column_stack([np.ones(x_others.shape[0]), x_others])
        beta, *_ = np.linalg.lstsq(x_others, y, rcond=None)
        y_hat = x_others @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / max(1e-12, (1.0 - r2))
        vif_records.append({"parameter": name, "vif": vif})
    return pd.DataFrame(vif_records)


def main() -> None:
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    participants, _ = load_all_participants(Config.DATA_DIR)
    matrices = build_analysis_matrices(participants)

    model_params = fit_all_models(matrices)
    model_params.to_csv(Config.OUTPUT_DIR / "og_model_params.csv", index=False)

    print("\nOG model fit results:")
    print(model_params.to_string(index=False))

    ai_params = model_params[model_params["model"] == "aiconf"][Config.PARAMETER_NAMES].values[0]
    self_params = model_params[model_params["model"] == "selfconf"][Config.PARAMETER_NAMES].values[0]
    pred_ai = compute_model_predictions_extended(ai_params, matrices["ai_conf_matrix"], matrices["e_tensor"])
    pred_self = compute_model_predictions_extended(self_params, matrices["self_conf_matrix"], matrices["e_tensor"])
    plot_conf(pred_ai, pred_self, matrices["ai_conf_matrix"], matrices["self_conf_matrix"], matrices["condition_array"])

    if Config.ROBUSTNESS_ENABLED:
        robust = run_robustness(matrices)
        robust_metrics = pd.DataFrame({
            "iteration": np.arange(1, robust["aiconf"]["mse"].shape[0] + 1),
            "aiconf_mse": robust["aiconf"]["mse"],
            "aiconf_r2": robust["aiconf"]["r2"],
            "aiconf_adj_r2": robust["aiconf"]["adj_r2"],
            "selfconf_mse": robust["selfconf"]["mse"],
            "selfconf_r2": robust["selfconf"]["r2"],
            "selfconf_adj_r2": robust["selfconf"]["adj_r2"],
        })
        robust_metrics.to_csv(Config.OUTPUT_DIR / "robust_metrics.csv", index=False)
        for model_name in ["aiconf", "selfconf"]:
            params_array = robust[model_name]["params"]
            df_r = pd.DataFrame(params_array, columns=Config.PARAMETER_NAMES)
            df_r.to_csv(Config.OUTPUT_DIR / f"robust_params_{model_name}.csv", index=False)
            if Config.VIF_ANALYSIS:
                vif_df = compute_vif(df_r)
                vif_df.to_csv(Config.OUTPUT_DIR / f"robust_param_vif_{model_name}.csv", index=False)


if __name__ == "__main__":
    main()

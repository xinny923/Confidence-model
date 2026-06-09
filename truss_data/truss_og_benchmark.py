"""
Benchmark the original Chong-style confidence model on the truss dataset.

This script mirrors the OG update equation used by Paper_fig/og_model.py,
adapted to the truss study's one-human/one-AI, 4-experience data:

    C(n+1) = C(n)
           + alpha_e * (E(n) - C(n))
           + alpha_a * (A(n) - C(n))
           + alpha_b * (B(n) - C(n))

    A(n) = gamma * C(n-1) + (1 - gamma) * A(n-1)
    E(n) = sum_i omega_i * e_i(n), i=1..4

Inputs are the MATLAB cell arrays in this folder:
    C_data.mat, sC_data.mat, e_data.mat, act_data.mat, feed*_data.mat, score*_data.mat

Outputs are written to truss_data/og_benchmark_output by default.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import least_squares


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    num_trials: int = 30
    change_trial: int = 20
    random_seed: int = 42
    max_nfev: int = 5000
    robustness_iterations: int = 100
    robustness_subset_per_condition: int = 40
    plot_dpi: int = 300


PARAMETER_NAMES = [
    "alpha_e",
    "alpha_a",
    "alpha_b",
    "omega1",
    "omega2",
    "omega3",
    "omega4",
    "gamma",
]

# Table 1 values from the truss paper are good starting points.
INITIAL_GUESS = {
    "aiconf": [0.329, 0.354, 0.0433, 0.870, 0.252, 0.117, 0.292, 0.299],
    "selfconf": [0.286, 0.275, 0.118, 0.623, 0.833, 0.195, 0.188, 0.279],
}

CONDITION_LABELS = {
    1: "hi-lo",
    2: "lo-hi",
}


def _load_first_variable(path: Path) -> Any:
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    keys = [key for key in mat if not key.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"Expected one MATLAB variable in {path}, found {keys}")
    return mat[keys[0]]


def _cell_to_conditions(cell: Any, expected_ndim: int, name: str) -> Dict[int, np.ndarray]:
    arr = np.asarray(cell)
    if arr.dtype != object or arr.size != 2:
        raise ValueError(f"{name} should be a 1x2 MATLAB cell array, got shape={arr.shape}, dtype={arr.dtype}")

    conditions: Dict[int, np.ndarray] = {}
    for cond, item in enumerate(arr.ravel(order="C"), start=1):
        values = np.asarray(item, dtype=float)
        if values.ndim != expected_ndim:
            raise ValueError(f"{name} condition {cond} expected {expected_ndim} dims, got {values.shape}")
        conditions[cond] = values
    return conditions


def load_truss_matrices(data_dir: Path) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    ai_by_cond = _cell_to_conditions(_load_first_variable(data_dir / "C_data.mat"), 2, "C_data")
    self_by_cond = _cell_to_conditions(_load_first_variable(data_dir / "sC_data.mat"), 2, "sC_data")
    e_by_cond = _cell_to_conditions(_load_first_variable(data_dir / "e_data.mat"), 3, "e_data")
    act_by_cond = _cell_to_conditions(_load_first_variable(data_dir / "act_data.mat"), 2, "act_data")
    feed1_by_cond = _cell_to_conditions(_load_first_variable(data_dir / "feed1_data.mat"), 2, "feed1_data")
    feed2_by_cond = _cell_to_conditions(_load_first_variable(data_dir / "feed2_data.mat"), 2, "feed2_data")
    score1_by_cond = _cell_to_conditions(_load_first_variable(data_dir / "score1_data.mat"), 2, "score1_data")
    score2_by_cond = _cell_to_conditions(_load_first_variable(data_dir / "score2_data.mat"), 2, "score2_data")

    ai_conf = []
    self_conf = []
    e_tensors = []
    act = []
    feed1 = []
    feed2 = []
    participant_rows = []
    pid = 1

    for cond in sorted(ai_by_cond):
        n_participants = ai_by_cond[cond].shape[0]
        for local_idx in range(n_participants):
            participant_rows.append({
                "pid": pid,
                "condition": cond,
                "condition_label": CONDITION_LABELS.get(cond, f"cond{cond}"),
                "condition_local_index": local_idx + 1,
                "individual_score": float(np.ravel(score1_by_cond[cond])[local_idx]),
                "team_score": float(np.ravel(score2_by_cond[cond])[local_idx]),
                "acceptance_rate": float(np.nanmean(act_by_cond[cond][local_idx])),
                "mean_aiconf": float(np.nanmean(ai_by_cond[cond][local_idx, 1:])),
                "mean_selfconf": float(np.nanmean(self_by_cond[cond][local_idx, 1:])),
            })
            pid += 1

        ai_conf.append(ai_by_cond[cond])
        self_conf.append(self_by_cond[cond])
        e_tensors.append(e_by_cond[cond])
        act.append(act_by_cond[cond])
        feed1.append(feed1_by_cond[cond])
        feed2.append(feed2_by_cond[cond])

    matrices = {
        "ai_conf_matrix": np.vstack(ai_conf),
        "self_conf_matrix": np.vstack(self_conf),
        "e_tensor": np.vstack(e_tensors),
        "act_matrix": np.vstack(act),
        "feed1_matrix": np.vstack(feed1),
        "feed2_matrix": np.vstack(feed2),
        "condition_array": np.concatenate([
            np.full(ai_by_cond[cond].shape[0], cond, dtype=int)
            for cond in sorted(ai_by_cond)
        ]),
        "pid_array": np.arange(1, sum(v.shape[0] for v in ai_by_cond.values()) + 1, dtype=int),
    }

    _validate_matrices(matrices)
    return matrices, pd.DataFrame(participant_rows)


def _validate_matrices(matrices: Dict[str, np.ndarray]) -> None:
    n_participants, series_len = matrices["ai_conf_matrix"].shape
    if series_len != Config.num_trials + 1:
        raise ValueError(f"Expected {Config.num_trials + 1} confidence points, got {series_len}")
    if matrices["self_conf_matrix"].shape != (n_participants, series_len):
        raise ValueError("Self-confidence matrix shape does not match AI-confidence matrix")
    if matrices["e_tensor"].shape != (n_participants, Config.num_trials, 4):
        raise ValueError(f"Expected e_tensor shape {(n_participants, Config.num_trials, 4)}, got {matrices['e_tensor'].shape}")
    if matrices["act_matrix"].shape != (n_participants, Config.num_trials):
        raise ValueError("Action matrix shape does not match participant/trial count")


def residuals_og(params: np.ndarray, observed: np.ndarray, e_tensor: np.ndarray) -> np.ndarray:
    predicted = predict_og(params, observed, e_tensor)
    residual = observed[:, 1:] - predicted[:, 1:]
    return residual.T.ravel()


def predict_og(params: np.ndarray, observed: np.ndarray, e_tensor: np.ndarray) -> np.ndarray:
    alpha_e, alpha_a, alpha_b, omega1, omega2, omega3, omega4, gamma = params
    n_participants, series_len = observed.shape
    n_trials = series_len - 1

    c = observed[:, 0].copy()
    a = observed[:, 0].copy()
    b = observed[:, 0].copy()
    predicted = np.zeros_like(observed, dtype=float)
    predicted[:, 0] = c

    for t in range(n_trials):
        if t > 0:
            # Match Code/myfun.m indexing: trial 2 still uses C0 in A, trial
            # 3 uses the prediction after trial 1, etc.
            a = gamma * predicted[:, t - 1] + (1.0 - gamma) * a
        experience = (
            omega1 * e_tensor[:, t, 0]
            + omega2 * e_tensor[:, t, 1]
            + omega3 * e_tensor[:, t, 2]
            + omega4 * e_tensor[:, t, 3]
        )
        c = c + alpha_e * (experience - c) + alpha_a * (a - c) + alpha_b * (b - c)
        predicted[:, t + 1] = np.clip(c, 0.0, 1.0)

    return predicted


def condition_mean_metric(
    observed: np.ndarray,
    predicted: np.ndarray,
    condition_array: np.ndarray,
    n_params: int,
) -> Dict[str, float]:
    obs_points = []
    pred_points = []
    within_condition_sstot = 0.0
    for cond in sorted(np.unique(condition_array)):
        idx = np.where(condition_array == cond)[0]
        obs_mean = np.nanmean(observed[idx, 1:], axis=0)
        pred_mean = np.nanmean(predicted[idx, 1:], axis=0)
        obs_points.append(obs_mean)
        pred_points.append(pred_mean)
        within_condition_sstot += float(np.nansum((obs_mean - np.nanmean(obs_mean)) ** 2))

    y = np.concatenate(obs_points)
    y_hat = np.concatenate(pred_points)
    ssres = float(np.nansum((y_hat - y) ** 2))
    global_sstot = float(np.nansum((y - np.nanmean(y)) ** 2))
    n_points = int(np.count_nonzero(~np.isnan(y)))
    r2 = 1.0 - ssres / within_condition_sstot if within_condition_sstot > 0 else float("nan")
    global_r2 = 1.0 - ssres / global_sstot if global_sstot > 0 else float("nan")
    # Match the MATLAB modelPlot.m / paper reporting convention:
    # condition-level SST is summed across the two mean curves, while adjusted R^2
    # uses the original participant/parameter factor 99/92.
    paper_adjustment = 99.0 / 92.0
    adj_r2 = 1.0 - paper_adjustment * (ssres / within_condition_sstot) if within_condition_sstot > 0 else float("nan")
    global_adj_r2 = (
        1.0 - ((n_points - 1) / max(1, n_points - n_params - 1)) * (ssres / global_sstot)
        if global_sstot > 0
        else float("nan")
    )
    return {
        "mse": ssres / n_points,
        "r2": r2,
        "adj_r2": adj_r2,
        "global_r2": global_r2,
        "global_adj_r2": global_adj_r2,
        "ssres": ssres,
        "sstot": within_condition_sstot,
        "global_sstot": global_sstot,
        "n_metric_points": n_points,
    }


def fit_model(
    model_name: str,
    observed: np.ndarray,
    e_tensor: np.ndarray,
    condition_array: np.ndarray,
) -> Dict[str, Any]:
    initial = np.array(INITIAL_GUESS[model_name], dtype=float)
    result = least_squares(
        residuals_og,
        x0=initial,
        bounds=(np.zeros_like(initial), np.ones_like(initial)),
        args=(observed, e_tensor),
        max_nfev=Config.max_nfev,
    )
    predicted = predict_og(result.x, observed, e_tensor)
    metrics = condition_mean_metric(observed, predicted, condition_array, len(PARAMETER_NAMES))
    metrics.update({
        "params": result.x,
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "message": result.message,
    })
    return metrics


def fit_all(matrices: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    records = []
    predictions = {}
    for model_name, matrix_name in [("aiconf", "ai_conf_matrix"), ("selfconf", "self_conf_matrix")]:
        fit = fit_model(model_name, matrices[matrix_name], matrices["e_tensor"], matrices["condition_array"])
        predicted = predict_og(fit["params"], matrices[matrix_name], matrices["e_tensor"])
        predictions[model_name] = predicted
        row = {
            "model": model_name,
            "mse": fit["mse"],
            "r2": fit["r2"],
            "adj_r2": fit["adj_r2"],
            "global_r2": fit["global_r2"],
            "global_adj_r2": fit["global_adj_r2"],
            "cost": fit["cost"],
            "nfev": fit["nfev"],
            "success": fit["success"],
        }
        row.update(dict(zip(PARAMETER_NAMES, fit["params"])))
        records.append(row)
    return pd.DataFrame(records), predictions


def trial_level_summary(matrices: Dict[str, np.ndarray], predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    tracks = [
        ("aiconf", matrices["ai_conf_matrix"], predictions["aiconf"]),
        ("selfconf", matrices["self_conf_matrix"], predictions["selfconf"]),
    ]
    for cond in sorted(np.unique(matrices["condition_array"])):
        idx = np.where(matrices["condition_array"] == cond)[0]
        for trial in range(1, Config.num_trials + 1):
            for track, observed, predicted in tracks:
                obs = float(np.nanmean(observed[idx, trial]))
                pred = float(np.nanmean(predicted[idx, trial]))
                rows.append({
                    "condition": int(cond),
                    "condition_label": CONDITION_LABELS.get(int(cond), f"cond{cond}"),
                    "trial": trial,
                    "track": track,
                    "mean_observed": obs,
                    "mean_predicted": pred,
                    "residual": obs - pred,
                    "abs_residual": abs(obs - pred),
                })
    return pd.DataFrame(rows)


def action_summary(matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for cond in sorted(np.unique(matrices["condition_array"])):
        idx = np.where(matrices["condition_array"] == cond)[0]
        acts = matrices["act_matrix"][idx]
        feed2 = matrices["feed2_matrix"][idx]
        for trial in range(Config.num_trials):
            rows.append({
                "condition": int(cond),
                "condition_label": CONDITION_LABELS.get(int(cond), f"cond{cond}"),
                "trial": trial + 1,
                "accept_rate": float(np.nanmean(acts[:, trial])),
                "positive_feedback_rate": float(np.nanmean(feed2[:, trial] > 0)),
            })
    return pd.DataFrame(rows)


def run_robustness(matrices: Dict[str, np.ndarray], n_iter: int, subset_per_condition: int) -> Dict[str, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(Config.random_seed)
    by_condition = {
        cond: np.where(matrices["condition_array"] == cond)[0]
        for cond in sorted(np.unique(matrices["condition_array"]))
    }
    results = {
        "aiconf": {"params": [], "mse": [], "r2": [], "adj_r2": []},
        "selfconf": {"params": [], "mse": [], "r2": [], "adj_r2": []},
    }

    for iteration in range(n_iter):
        if (iteration + 1) % max(1, n_iter // 10) == 0:
            logger.info("Robustness progress: %s/%s", iteration + 1, n_iter)
        sample_idx = []
        for cond, idx in by_condition.items():
            k = min(subset_per_condition, len(idx))
            sample_idx.extend(rng.choice(idx, size=k, replace=False).tolist())
        sample_idx = np.array(sorted(sample_idx), dtype=int)
        e_sub = matrices["e_tensor"][sample_idx]
        cond_sub = matrices["condition_array"][sample_idx]
        for model_name, matrix_name in [("aiconf", "ai_conf_matrix"), ("selfconf", "self_conf_matrix")]:
            fit = fit_model(model_name, matrices[matrix_name][sample_idx], e_sub, cond_sub)
            results[model_name]["params"].append(fit["params"])
            results[model_name]["mse"].append(fit["mse"])
            results[model_name]["r2"].append(fit["r2"])
            results[model_name]["adj_r2"].append(fit["adj_r2"])

    for model_name in results:
        for key in results[model_name]:
            results[model_name][key] = np.asarray(results[model_name][key], dtype=float)
    return results


def summarize_robustness(robust: Dict[str, Dict[str, np.ndarray]], params: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, data in robust.items():
        original = params[params["model"] == model_name].iloc[0]
        for i, param_name in enumerate(PARAMETER_NAMES):
            values = data["params"][:, i]
            rows.append({
                "model": model_name,
                "parameter": param_name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "original": float(original[param_name]),
            })
        for metric in ["mse", "r2", "adj_r2"]:
            values = data[metric]
            rows.append({
                "model": model_name,
                "parameter": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "original": float(original[metric]),
            })
    return pd.DataFrame(rows)


def save_robustness_outputs(robust: Dict[str, Dict[str, np.ndarray]], params: pd.DataFrame, output_dir: Path) -> None:
    metrics = pd.DataFrame({
        "iteration": np.arange(1, robust["aiconf"]["mse"].shape[0] + 1),
        "aiconf_mse": robust["aiconf"]["mse"],
        "aiconf_r2": robust["aiconf"]["r2"],
        "aiconf_adj_r2": robust["aiconf"]["adj_r2"],
        "selfconf_mse": robust["selfconf"]["mse"],
        "selfconf_r2": robust["selfconf"]["r2"],
        "selfconf_adj_r2": robust["selfconf"]["adj_r2"],
    })
    metrics.to_csv(output_dir / "robust_metrics.csv", index=False)
    for model_name, data in robust.items():
        pd.DataFrame(data["params"], columns=PARAMETER_NAMES).to_csv(
            output_dir / f"robust_params_{model_name}.csv",
            index=False,
        )
    summarize_robustness(robust, params).to_csv(output_dir / "robustness_statistics.csv", index=False)


def plot_condition_fit(matrices: Dict[str, np.ndarray], predictions: Dict[str, np.ndarray], output_dir: Path) -> None:
    trials = np.arange(0, Config.num_trials + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)

    for col, cond in enumerate(sorted(np.unique(matrices["condition_array"]))):
        idx = np.where(matrices["condition_array"] == cond)[0]
        first_label, second_label = ("80% AI", "20% AI") if cond == 1 else ("20% AI", "80% AI")
        for row, (track, observed_key, ylabel) in enumerate([
            ("aiconf", "ai_conf_matrix", "Confidence in AI"),
            ("selfconf", "self_conf_matrix", "Self-confidence"),
        ]):
            ax = axes[row, col]
            observed = matrices[observed_key][idx]
            predicted = predictions[track][idx]
            mean_obs = np.nanmean(observed, axis=0)
            se_obs = np.nanstd(observed, axis=0, ddof=1) / np.sqrt(observed.shape[0])
            mean_pred = np.nanmean(predicted, axis=0)

            ax.axvspan(0, Config.change_trial, ymin=0.0, ymax=0.11, color="#cfe1f2", alpha=0.8)
            ax.axvspan(Config.change_trial, Config.num_trials, ymin=0.0, ymax=0.11, color="#f6dfb8", alpha=0.8)
            ax.text(Config.change_trial / 2, 0.05, first_label, ha="center", va="center", fontsize=9)
            ax.text((Config.change_trial + Config.num_trials) / 2, 0.05, second_label, ha="center", va="center", fontsize=9)
            ax.axvline(Config.change_trial, color="#ff7f0e", linewidth=1.3)
            ax.errorbar(trials, mean_obs, yerr=se_obs, fmt="o", color="black", markersize=4, capsize=2, label="Data")
            ax.plot(trials, mean_pred, color="#1f77b4", linewidth=2.2, label="OG fit")
            ax.set_ylim(0, 1)
            ax.set_xlim(0, Config.num_trials)
            ax.grid(alpha=0.3)
            ax.set_title(f"{CONDITION_LABELS.get(int(cond), cond)} / {track}")
            ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Trial")
            if row == 0 and col == 0:
                ax.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "figure_condition_fit.png", dpi=Config.plot_dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the OG confidence model on truss_data.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "og_benchmark_output")
    parser.add_argument("--no-robustness", action="store_true", help="Skip 80-participant robustness benchmark.")
    parser.add_argument("--robustness-iterations", type=int, default=Config.robustness_iterations)
    parser.add_argument("--subset-per-condition", type=int, default=Config.robustness_subset_per_condition)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrices, participants = load_truss_matrices(args.data_dir)
    participants.to_csv(args.output_dir / "participant_summary.csv", index=False)

    params, predictions = fit_all(matrices)
    params.to_csv(args.output_dir / "og_model_params.csv", index=False)
    trial_level_summary(matrices, predictions).to_csv(args.output_dir / "trial_level_fit.csv", index=False)
    action_summary(matrices).to_csv(args.output_dir / "action_feedback_summary.csv", index=False)
    plot_condition_fit(matrices, predictions, args.output_dir)

    print("\nTruss OG model benchmark:")
    print(params.to_string(index=False))

    if not args.no_robustness:
        robust = run_robustness(matrices, args.robustness_iterations, args.subset_per_condition)
        save_robustness_outputs(robust, params, args.output_dir)

    logger.info("Saved truss OG benchmark outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()

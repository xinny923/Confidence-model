"""
Benchmark_rigor: strict reproduction of original MATLAB model (2 actions, 4 experiences).
- Binary experience (e1-e4) only, no soft weights.
- Mean-data R^2 with factor 99/92, MSE = ssres/60 (2 cond x 30 trials).
- Robustness: 100 iterations, 80 participants each.
Outputs to output_benchmark_rigor/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import least_squares
import statsmodels.formula.api as smf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Config:
    NUM_TRIALS = 30
    PRACTICE_LAST_IDX = 4
    MAIN_START_IDX = 6
    SCORE_ROW_IDX = MAIN_START_IDX + NUM_TRIALS

    PARAMETER_NAMES = [
        "alpha_e", "alpha_a", "alpha_b",
        "omega1", "omega2", "omega3", "omega4",
        "gamma",
    ]

    INITIAL = {
        "aiconf": [0.2672, 0.3405, 0.0524, 0.8439, 0.2115, 0.0, 0.5217, 0.3897],
        "selfconf": [0.2844, 0.4706, 0.0, 0.5736, 0.8284, 0.2384, 0.2863, 0.1147],
    }

    DATA_DIR = Path("Data")
    CODE_DIR = Path("Code")
    OUTPUT_DIR = Path("output_benchmark_rigor")
    PLOT_DPI = 300
    RANDOM_SEED = 42
    ROBUSTNESS_N_ITER = 100
    ROBUSTNESS_SUBSET = 80
    PERF_CHANGE_TRIAL = 20


@dataclass
class ParticipantData:
    pid: int
    base_id: int
    condition: int
    c_series: np.ndarray
    self_series: np.ndarray
    e_matrix: np.ndarray  # (30,4)
    act_series: np.ndarray
    perf_series: np.ndarray
    skill_score: float
    team_score: float
    trial_records: List[Dict[str, Any]]


def safe_float(value: Any, default: float = math.nan) -> float:
    """Convert to float; return default on bad inputs."""
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
    m = re.search(r"data(\d+)_(\d+)\.csv", path.name)
    if not m:
        raise ValueError(f"Invalid filename: {path.name}")
    return int(m.group(1)), int(m.group(2))


def parse_participant(path: Path, pid: int, condition: int, base_id: int) -> ParticipantData:
    df = pd.read_csv(path, header=None)

    def get_num(r: int, c: int) -> float:
        return safe_float(df.iat[r, c])

    def get_str(r: int, c: int) -> str:
        return safe_str(df.iat[r, c])

    # Initialize full-length arrays; we will skip invalid trials
    c_series = [get_num(Config.PRACTICE_LAST_IDX, 9)]
    self_series = [get_num(Config.PRACTICE_LAST_IDX, 8)]
    e_matrix = []
    act_values = [0]
    trial_records: List[Dict[str, Any]] = []

    valid_trials = 0
    for t in range(Config.NUM_TRIALS):
        r = Config.MAIN_START_IDX + t
        ai_suggestion = get_str(r, 3)
        final_move = get_str(r, 5)
        feedback2 = get_num(r, 7)
        selfconf = get_num(r, 8)
        aiconf = get_num(r, 9)

        # Clamp confidences to [0,1]
        def clamp01(x): 
            return min(max(x, 0.0), 1.0) if not math.isnan(x) else math.nan
        selfconf = clamp01(selfconf)
        aiconf = clamp01(aiconf)

        # Skip trial if key fields missing
        if math.isnan(feedback2) or math.isnan(selfconf) or math.isnan(aiconf):
            continue

        accept = int(ai_suggestion == final_move)
        row_e = np.zeros(4, dtype=float)
        sign = int(feedback2 / 5 * -1)
        idx = (sign + 1) if accept else (sign + 2)
        if 0 <= idx < 4:
            row_e[idx] = 1.0

        c_series.append(aiconf)
        self_series.append(selfconf)
        e_matrix.append(row_e)
        act_values.append(accept)
        valid_trials += 1

        trial_records.append({
            "pid": pid,
            "condition": condition,
            "trial": valid_trials,
            "feedback2": feedback2,
            "selfconf": selfconf,
            "aiconf": aiconf,
            "accept": accept,
        })

    # Pad to 30 if needed (rare; in case of skipped trials) with NaN so shapes align
    while len(c_series) < Config.NUM_TRIALS + 1:
        c_series.append(math.nan)
        self_series.append(math.nan)
        e_matrix.append(np.zeros(4, dtype=float))
        act_values.append(math.nan)

    c_series = np.array(c_series, dtype=float)
    self_series = np.array(self_series, dtype=float)
    e_matrix = np.array(e_matrix, dtype=float)
    act_values = np.array(act_values, dtype=float)

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
        act_series=act_values,
        perf_series=perf_series,
        skill_score=skill_score,
        team_score=team_score,
        trial_records=trial_records,
    )


def load_all_participants(data_dir: Path) -> Tuple[List[ParticipantData], pd.DataFrame]:
    file_map: Dict[int, Dict[int, Path]] = {1: {}, 2: {}}
    for p in sorted(data_dir.glob("data*_*.csv")):
        base_id, cond = parse_filename(p)
        file_map[cond][base_id] = p
    participants = []
    records = []
    pid = 1
    for cond in (1, 2):
        for base_id in sorted(file_map[cond]):
            part = parse_participant(file_map[cond][base_id], pid, cond, base_id)
            participants.append(part)
            records.extend(part.trial_records)
            pid += 1
    return participants, pd.DataFrame(records)


def build_matrices(parts: List[ParticipantData]) -> Dict[str, np.ndarray]:
    return {
        "ai_conf_matrix": np.vstack([p.c_series for p in parts]),
        "self_conf_matrix": np.vstack([p.self_series for p in parts]),
        "e_tensor": np.stack([p.e_matrix for p in parts]),
        "act_matrix": np.vstack([p.act_series for p in parts]),
        "pid_array": np.array([p.pid for p in parts], dtype=int),
        "condition_array": np.array([p.condition for p in parts], dtype=int),
        "team_scores": np.array([p.team_score for p in parts], dtype=float),
        "individual_scores": np.array([p.skill_score for p in parts], dtype=float),
    }


def simulate_confidence(params: np.ndarray, observed: np.ndarray, e_tensor: np.ndarray) -> np.ndarray:
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
            a = gamma * predicted[:, t] + (1 - gamma) * a
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


def fit_model(observed: np.ndarray, e_tensor: np.ndarray, initial: np.ndarray) -> Dict[str, Any]:
    bounds = (np.zeros_like(initial), np.ones_like(initial))
    result = least_squares(
        simulate_confidence,
        x0=initial,
        bounds=bounds,
        args=(observed, e_tensor),
        max_nfev=5000,
    )
    # mean-data R2
    # Use only valid (non-NaN) trials for mean-data R2
    pred_all = compute_predictions(result.x, observed, e_tensor)
    valid_mask = ~np.isnan(observed)
    mean_obs = np.nanmean(observed, axis=0)
    mean_pred = np.nanmean(pred_all, axis=0)
    # trials 1..end and non-NaN
    mean_obs_trials = mean_obs[1:]
    mean_pred_trials = mean_pred[1:]
    ssres = np.nansum((mean_pred_trials - mean_obs_trials) ** 2)
    sstot = np.nansum((mean_obs_trials - np.nanmean(mean_obs_trials)) ** 2)
    adj_r2 = 1 - (99 / 92) * (ssres / sstot) if sstot > 0 else float("nan")
    mse_mean = ssres / (2 * Config.NUM_TRIALS)
    residuals = simulate_confidence(result.x, observed, e_tensor)
    return {"params": result.x, "mse": mse_mean, "adj_r2": adj_r2, "residuals": residuals}


def compute_predictions(params: np.ndarray, observed: np.ndarray, e_tensor: np.ndarray) -> np.ndarray:
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
            a = gamma * predicted[:, t] + (1 - gamma) * a
        experience = (
            omega1 * e_tensor[:, t, 0] +
            omega2 * e_tensor[:, t, 1] +
            omega3 * e_tensor[:, t, 2] +
            omega4 * e_tensor[:, t, 3]
        )
        c = c + alpha_e * (experience - c) + alpha_a * (a - c) + alpha_b * (b - c)
        predicted[:, t + 1] = c
    return predicted


def load_score_categories(code_dir: Path) -> Dict[str, List[int]]:
    categories = {"poor": [], "fair": [], "good": []}
    offsets = {1: 0, 2: 50}
    for cond in (1, 2):
        mat = loadmat(code_dir / f"scoregroup{cond}.mat")["group"][0]
        for cat_name, arr in zip(["poor", "fair", "good"], mat):
            ids = [int(x) for x in arr.flatten()]
            categories[cat_name].extend([offsets[cond] + i for i in ids])
    return categories


def prepare_logit(matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
    recs = []
    pid = matrices["pid_array"]
    ai_conf = matrices["ai_conf_matrix"][:, 1:]
    self_conf = matrices["self_conf_matrix"][:, 1:]
    accept = matrices["act_matrix"][:, 1:]
    for i, participant_id in enumerate(pid):
        for t in range(Config.NUM_TRIALS):
            recs.append({
                "pid": participant_id,
                "trial": t + 1,
                "aiconf": ai_conf[i, t],
                "selfconf": self_conf[i, t],
                "accept": accept[i, t],
            })
    return pd.DataFrame(recs)


def run_overall_logit(logit_df: pd.DataFrame) -> pd.DataFrame:
    df = logit_df.copy()
    model = smf.logit("accept ~ aiconf + selfconf + C(pid)", data=df).fit(method="lbfgs", maxiter=1000, disp=False)
    return pd.DataFrame({
        "predictor": ["aiconf", "selfconf"],
        "coef": [model.params["aiconf"], model.params["selfconf"]],
        "std_err": [model.bse["aiconf"], model.bse["selfconf"]],
        "p_value": [model.pvalues["aiconf"], model.pvalues["selfconf"]],
    })


def run_category_logit(logit_df: pd.DataFrame, score_groups: Dict[str, List[int]]) -> pd.DataFrame:
    records = []
    for category in ["poor", "fair", "good"]:
        subset = logit_df[logit_df["pid"].isin(score_groups[category])]
        if subset.empty:
            continue
        model = smf.logit("accept ~ aiconf + selfconf + C(pid)", data=subset).fit(method="lbfgs", maxiter=1000, disp=False)
        for predictor in ("aiconf", "selfconf"):
            records.append({
                "category": category,
                "predictor": predictor,
                "coef": model.params[predictor],
                "std_err": model.bse[predictor],
                "p_value": model.pvalues[predictor],
            })
    return pd.DataFrame(records)


def run_robustness(matrices: Dict[str, np.ndarray]) -> Dict[str, Any]:
    np.random.seed(Config.RANDOM_SEED)
    n = matrices["ai_conf_matrix"].shape[0]
    subset = min(Config.ROBUSTNESS_SUBSET, n)
    results = {
        "aiconf": {"params": [], "mse": [], "r2_adj": []},
        "selfconf": {"params": [], "mse": [], "r2_adj": []},
    }
    for _ in range(Config.ROBUSTNESS_N_ITER):
        idx = np.random.choice(n, size=subset, replace=False)
        idx.sort()
        e_sub = matrices["e_tensor"][idx]
        for name, mat in [("aiconf", matrices["ai_conf_matrix"]), ("selfconf", matrices["self_conf_matrix"])]:
            obs = mat[idx]
            initial = np.array(Config.INITIAL[name], dtype=float)
            fit_res = fit_model(obs, e_sub, initial)
            results[name]["params"].append(fit_res["params"])
            results[name]["mse"].append(fit_res["mse"])
            results[name]["r2_adj"].append(fit_res["adj_r2"])
    for key in results:
        results[key]["params"] = np.array(results[key]["params"])
        results[key]["mse"] = np.array(results[key]["mse"])
        results[key]["r2_adj"] = np.array(results[key]["r2_adj"])
    return results


def summarize_robustness(robust: Dict[str, Any], base_params: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name in ["aiconf", "selfconf"]:
        params_array = robust[name]["params"]
        mse_array = robust[name]["mse"]
        r2_array = robust[name]["r2_adj"]
        base = base_params[base_params["model"] == name].iloc[0]
        for i, p_name in enumerate(Config.PARAMETER_NAMES):
            vals = params_array[:, i]
            records.append({
                "model": name,
                "parameter": p_name,
                "mean": vals.mean(),
                "std": vals.std(ddof=1),
                "cv": vals.std(ddof=1) / (abs(vals.mean()) + 1e-10),
                "q25": np.percentile(vals, 25),
                "q75": np.percentile(vals, 75),
                "min": vals.min(),
                "max": vals.max(),
                "original": base[p_name],
            })
        records.append({"model": name, "parameter": "mse", "mean": mse_array.mean(), "std": mse_array.std(ddof=1), "cv": mse_array.std(ddof=1)/(mse_array.mean()+1e-10), "original": np.nan})
        records.append({"model": name, "parameter": "r2_adj", "mean": r2_array.mean(), "std": r2_array.std(ddof=1), "cv": r2_array.std(ddof=1)/(abs(r2_array.mean())+1e-10), "original": np.nan})
    return pd.DataFrame(records)


def plot_figure3(model_params: pd.DataFrame, matrices: Dict[str, np.ndarray], output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    ai_params = model_params[model_params["model"] == "aiconf"].iloc[0][Config.PARAMETER_NAMES].values
    self_params = model_params[model_params["model"] == "selfconf"].iloc[0][Config.PARAMETER_NAMES].values
    cond_mask = {1: matrices["condition_array"] == 1, 2: matrices["condition_array"] == 2}
    trials = np.arange(1, Config.NUM_TRIALS + 1)
    change_point = Config.PERF_CHANGE_TRIAL

    def add_perf_bands(ax: plt.Axes, cond: int) -> None:
        if cond == 1:
            ax.axvspan(1, change_point, color="#cfe1f2", alpha=0.5, zorder=0)
            ax.axvspan(change_point, Config.NUM_TRIALS, color="#f7e0b5", alpha=0.5, zorder=0)
            ax.text(6, 0.08, "High-performing AI\n(80% accuracy)", fontsize=9)
            ax.text(22, 0.08, "Low-performing AI\n(20% accuracy)", fontsize=9)
        else:
            ax.axvspan(1, change_point, color="#f7e0b5", alpha=0.5, zorder=0)
            ax.axvspan(change_point, Config.NUM_TRIALS, color="#cfe1f2", alpha=0.5, zorder=0)
            ax.text(6, 0.08, "Low-performing AI\n(20% accuracy)", fontsize=9)
            ax.text(22, 0.08, "High-performing AI\n(80% accuracy)", fontsize=9)

    for col, cond in enumerate([1, 2]):
        mask = cond_mask[cond]
        e_sub = matrices["e_tensor"][mask]

        ai_obs = matrices["ai_conf_matrix"][mask][:, 1:]
        ai_pred = compute_predictions(ai_params, matrices["ai_conf_matrix"][mask], e_sub)[:, 1:]
        ai_mean = np.nanmean(ai_obs, axis=0)
        ai_se = np.nanstd(ai_obs, axis=0, ddof=1) / np.sqrt(ai_obs.shape[0])
        ai_pred_mean = np.nanmean(ai_pred, axis=0)

        ax_ai = axes[0, col]
        add_perf_bands(ax_ai, cond)
        vline = ax_ai.axvline(change_point, color="orange", linewidth=1.5, label="AI performance change")
        ax_ai.errorbar(trials, ai_mean, yerr=ai_se, fmt="o", color="black", markersize=4, capsize=2, label="Data")
        ax_ai.plot(trials, ai_pred_mean, color="#1f77b4", linewidth=2.5, label="Average model fit")
        ax_ai.set_ylim(0, 1)
        ax_ai.set_xlim(1, Config.NUM_TRIALS)
        ax_ai.grid(alpha=0.3)
        ax_ai.set_title("Confidence in AI")
        ax_ai.text(0.02, 0.98, "A" if col == 0 else "B", transform=ax_ai.transAxes,
                   fontsize=12, fontweight="bold", va="top")
        ax_ai.legend(handles=[ax_ai.lines[-1], vline, ax_ai.lines[0]], loc="upper right", fontsize=9)

        self_obs = matrices["self_conf_matrix"][mask][:, 1:]
        self_pred = compute_predictions(self_params, matrices["self_conf_matrix"][mask], e_sub)[:, 1:]
        self_mean = np.nanmean(self_obs, axis=0)
        self_se = np.nanstd(self_obs, axis=0, ddof=1) / np.sqrt(self_obs.shape[0])
        self_pred_mean = np.nanmean(self_pred, axis=0)

        ax_self = axes[1, col]
        add_perf_bands(ax_self, cond)
        vline = ax_self.axvline(change_point, color="orange", linewidth=1.5, label="AI performance change")
        ax_self.errorbar(trials, self_mean, yerr=self_se, fmt="o", color="black", markersize=4, capsize=2, label="Data")
        ax_self.plot(trials, self_pred_mean, color="#1f77b4", linewidth=2.5, label="Average model fit")
        ax_self.set_ylim(0, 1)
        ax_self.set_xlim(1, Config.NUM_TRIALS)
        ax_self.grid(alpha=0.3)
        ax_self.set_title("Self-confidence")
        ax_self.text(0.02, 0.98, "C" if col == 0 else "D", transform=ax_self.transAxes,
                     fontsize=12, fontweight="bold", va="top")
        ax_self.set_xlabel("Puzzle number, n")
        ax_self.legend(handles=[ax_self.lines[-1], vline, ax_self.lines[0]], loc="upper right", fontsize=9)

    axes[0, 0].set_ylabel("Confidence in AI")
    axes[1, 0].set_ylabel("Self-confidence")

    fig.tight_layout()
    fig.savefig(output_dir / "figure3_model_fitting.png", dpi=Config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_figure4(matrices: Dict[str, np.ndarray], output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cond_mask = {1: matrices["condition_array"] == 1, 2: matrices["condition_array"] == 2}
    change_point = Config.PERF_CHANGE_TRIAL
    trials = np.arange(1, Config.NUM_TRIALS + 1)
    for row, cond in enumerate([1, 2]):
        mask = cond_mask[cond]
        ai_conf = matrices["ai_conf_matrix"][mask, 1:]
        self_conf = matrices["self_conf_matrix"][mask, 1:]
        for col, (conf, title) in enumerate([(ai_conf, "AI"), (self_conf, "Self")]):
            mean = np.nanmean(conf, axis=0)
            se = np.nanstd(conf, axis=0, ddof=1) / np.sqrt(conf.shape[0])
            ax = axes[row, col]
            ax.errorbar(trials, mean, yerr=se, fmt='o', color='black', markersize=4, capsize=2, alpha=0.5)
            t1 = trials[:change_point]; t2 = trials[change_point:]
            if len(t1) > 1:
                coef1 = np.polyfit(t1, mean[:change_point], 1); ax.plot(t1, np.poly1d(coef1)(t1), color='#1f77b4', linewidth=2.5)
            if len(t2) > 1:
                coef2 = np.polyfit(t2, mean[change_point:], 1); ax.plot(t2, np.poly1d(coef2)(t2), color='#1f77b4', linewidth=2.5)
            ax.axvline(change_point, color='orange', linestyle='--', linewidth=2, alpha=0.7)
            ax.set_ylim(0, 1); ax.grid(alpha=0.3)
            ax.set_title(f"{title} Confidence (Cond {cond})"); ax.set_xlabel("Trial"); ax.set_ylabel("Confidence")
    fig.tight_layout()
    fig.savefig(output_dir / "figure4_confidence_trends.png", dpi=Config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_figure5(scores: pd.DataFrame, score_groups: Dict[str, List[int]], output_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for ax, (cond, group) in zip(axes, scores.groupby("condition")):
        ax.hist(group["team_score"], bins=15, alpha=0.8, color="#4c72b0", edgecolor="black", linewidth=0.5)
        sorted_scores = np.sort(group["team_score"].values)
        n = len(sorted_scores); idx_25 = int(0.25 * n); idx_75 = int(0.75 * n)
        lower, upper = sorted_scores[idx_25], sorted_scores[idx_75]
        ax.axvline(lower, color='orange', linestyle='--', linewidth=2)
        ax.axvline(upper, color='orange', linestyle='--', linewidth=2)
        ax.set_title(f"Condition {cond}"); ax.set_xlabel("Team Performance Score"); ax.set_ylabel("Number of Participants"); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(output_dir / "figure5_score_distributions.png", dpi=Config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_figure6(scores: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    color_map = {"poor": "#d62728", "fair": "#ff7f0e", "good": "#2ca02c"}
    for category, subset in scores.groupby("category"):
        color = color_map.get(category, "#1f77b4")
        axes[0].scatter(subset["team_score"], subset["individual_score"], label=category.capitalize(),
                        color=color, alpha=0.7, s=80, edgecolors="black", linewidths=0.8)
        axes[1].scatter(subset["team_score"], subset["mean_selfconf"], label=category.capitalize(),
                        color=color, alpha=0.7, s=80, edgecolors="black", linewidths=0.8)
    axes[0].set_xlabel("Team Performance Score"); axes[0].set_ylabel("Individual Skill Score")
    axes[0].set_title("(A) Team vs Individual Performance"); axes[0].legend(frameon=True, shadow=True); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Team Performance Score"); axes[1].set_ylabel("Average Self-confidence")
    axes[1].set_title("(B) Team Score vs Self-confidence"); axes[1].legend(frameon=True, shadow=True); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "figure6_scatter_plots.png", dpi=Config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    participants, trial_df = load_all_participants(Config.DATA_DIR)
    matrices = build_matrices(participants)
    trial_df.to_csv(Config.OUTPUT_DIR / "trial_data_rigor.csv", index=False)

    # participant summary
    records = []
    for p in participants:
        main_slice = slice(1, None)
        records.append({
            "pid": p.pid,
            "condition": p.condition,
            "acceptance_rate": np.nanmean(p.act_series[main_slice]),
            "team_score": p.team_score,
            "individual_score": p.skill_score,
            "mean_selfconf": np.nanmean(p.self_series[main_slice]),
            "mean_aiconf": np.nanmean(p.c_series[main_slice]),
        })
    scores = pd.DataFrame(records)
    score_groups = load_score_categories(Config.CODE_DIR)
    scores["category"] = "fair"
    for cat in ("poor", "good"):
        scores.loc[scores["pid"].isin(score_groups[cat]), "category"] = cat
    scores.to_csv(Config.OUTPUT_DIR / "participant_summary_rigor.csv", index=False)

    # Fit models (full sample)
    records = []
    for name, conf_matrix in [("aiconf", matrices["ai_conf_matrix"]), ("selfconf", matrices["self_conf_matrix"])]:
        initial = np.array(Config.INITIAL[name], dtype=float)
        fit_res = fit_model(conf_matrix, matrices["e_tensor"], initial)
        rec = {"model": name, "mse": fit_res["mse"], "r2_adj": fit_res["adj_r2"]}
        rec.update(dict(zip(Config.PARAMETER_NAMES, fit_res["params"])))
        records.append(rec)
    model_params = pd.DataFrame(records)
    model_params.to_csv(Config.OUTPUT_DIR / "table1_model_params_rigor.csv", index=False)

    # Figures
    plot_figure3(model_params, matrices, Config.OUTPUT_DIR)
    plot_figure4(matrices, Config.OUTPUT_DIR)
    plot_figure5(scores, score_groups, Config.OUTPUT_DIR)
    plot_figure6(scores, Config.OUTPUT_DIR)

    # Logistic regression
    logit_df = prepare_logit(matrices)
    overall = run_overall_logit(logit_df); overall.to_csv(Config.OUTPUT_DIR / "table2_logit_coeffs_rigor.csv", index=False)
    category = run_category_logit(logit_df, score_groups); category.to_csv(Config.OUTPUT_DIR / "table3_category_logit_rigor.csv", index=False)

    # Robustness
    robust = run_robustness(matrices)
    robust_stats = summarize_robustness(robust, model_params)
    robust_stats.to_csv(Config.OUTPUT_DIR / "robustness_statistics_rigor.csv", index=False)
    final_params = []
    for name in ["aiconf", "selfconf"]:
        final_params.append({
            "model": name,
            "mse": robust[name]["mse"].mean(),
            "r2_adj": robust[name]["r2_adj"].mean(),
            **dict(zip(Config.PARAMETER_NAMES, robust[name]["params"].mean(axis=0)))
        })
    pd.DataFrame(final_params).to_csv(Config.OUTPUT_DIR / "final_robust_params_rigor.csv", index=False)

    print("\n=== Rigor benchmark (MATLAB-style) ===")
    print(model_params.to_string(index=False))
    print("\nRobust mean params:")
    print(pd.DataFrame(final_params).to_string(index=False))


if __name__ == "__main__":
    main()

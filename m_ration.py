"""
Rational expectation weighting (E = V_raw - gamma * M) using softmax M.
4-term model, outputs to output_ration/, with fig3-6 and robustness.
"""

from __future__ import annotations

import logging
import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy import stats
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
        "gamma"
    ]

    INITIAL = {
        "aiconf": [0.2672, 0.3405, 0.0524, 0.8439, 0.2115, 0.0, 0.5217, 0.3897],
        "selfconf": [0.2844, 0.4706, 0.0, 0.5736, 0.8284, 0.2384, 0.2863, 0.1147],
    }

    OUTPUT_DIR = Path("output_ration")
    DATA_DIR = Path("Data")
    CODE_DIR = Path("Code")
    PLOT_DPI = 300
    RANDOM_SEED = 42
    ROBUSTNESS_N_ITER = 100
    ROBUSTNESS_SUBSET = 80
    PERF_CHANGE_TRIAL = 20
    GAMMA_COST = 1.0  # modify cost weight for M


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


def softmax_probs(logits: List[float]) -> List[float]:
    arr = np.array(logits, dtype=float)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return (exp / exp.sum()).tolist()


def parse_participant(path: Path, pid: int, condition: int, base_id: int) -> ParticipantData:
    df = pd.read_csv(path, header=None)

    def get_num(r, c):
        return safe_float(df.iat[r, c])

    def get_str(r, c):
        return safe_str(df.iat[r, c])

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
    act_series = [0]
    trial_records: List[Dict[str, Any]] = []

    for t in range(Config.NUM_TRIALS):
        r = Config.MAIN_START_IDX + t
        original_move = get_str(r, 2)
        ai_suggestion = get_str(r, 3)
        final_move = get_str(r, 5)
        feedback2 = get_num(r, 7)
        logits_moves_raw = safe_str(df.iat[r, 1])
        logits: List[float] = []
        moves: List[str] = []
        try:
            parsed = ast.literal_eval(logits_moves_raw)
            if isinstance(parsed, list) and len(parsed) == 2:
                logits, moves = parsed
        except Exception:
            pass

        # softmax probability for final_move
        miu = 0.0
        max_miu = 1.0
        if logits and moves:
            probs = softmax_probs(logits)
            max_miu = max(probs)
            move_to_prob = {m: p for m, p in zip(moves, probs)}
            miu = move_to_prob.get(final_move, 0.0)
        M = 1 - miu / max_miu if max_miu > 0 else 1.0
        M = float(np.clip(M, 0.0, 1.0))

        action_label = "accept" if final_move == ai_suggestion else ("reject" if final_move == original_move else "modify")
        accept = 1 if final_move == ai_suggestion else 0
        act_series.append(accept)

        if not math.isnan(feedback2):
            if action_label == "modify":
                V_raw = max_miu  # highest AI prob
                C_mod = Config.GAMMA_COST * M
                E_eff = V_raw - C_mod
                # positive evidence -> e1, negative -> e3
                if E_eff > 0:
                    e_matrix[t, 0] = E_eff
                elif E_eff < 0:
                    e_matrix[t, 2] = abs(E_eff)
            else:
                sign = int(feedback2 / 5 * -1)
                idx = (sign + 1) if accept else (sign + 2)
                if 0 <= idx < 4:
                    e_matrix[t, idx] = 1.0

        trial_records.append({
            "pid": pid,
            "condition": condition,
            "trial": t + 1,
            "action_label": action_label,
            "feedback2": feedback2,
            "selfconf": get_num(r, 8),
            "aiconf": get_num(r, 9),
            "modify_magnitude": M,
            "orig_eq_ai": int(original_move == ai_suggestion),
        })

    perf_series = np.array(
        [get_num(Config.PRACTICE_LAST_IDX, 7)] +
        [get_num(Config.MAIN_START_IDX + t, 7) for t in range(Config.NUM_TRIALS)],
        dtype=float
    )
    skill_score = np.nansum([get_num(Config.MAIN_START_IDX + t, 6) for t in range(Config.NUM_TRIALS)])
    team_score = get_num(Config.SCORE_ROW_IDX, 1)

    return ParticipantData(
        pid=pid, base_id=base_id, condition=condition,
        c_series=c_series, self_series=self_series, e_matrix=e_matrix,
        act_series=np.array(act_series, dtype=float), perf_series=perf_series,
        skill_score=skill_score, team_score=team_score, trial_records=trial_records
    )


def load_all_participants() -> Tuple[List[ParticipantData], pd.DataFrame]:
    file_map: Dict[int, Dict[int, Path]] = {1: {}, 2: {}}
    for p in sorted(Config.DATA_DIR.glob("data*_*.csv")):
        base_id, cond = parse_filename(p)
        file_map[cond][base_id] = p
    participants: List[ParticipantData] = []
    trial_records: List[Dict[str, Any]] = []
    pid_counter = 1
    for cond in (1, 2):
        for base_id in sorted(file_map[cond]):
            part = parse_participant(file_map[cond][base_id], pid_counter, cond, base_id)
            participants.append(part)
            trial_records.extend(part.trial_records)
            pid_counter += 1
    return participants, pd.DataFrame(trial_records)


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


def fit_model(observed: np.ndarray, e_tensor: np.ndarray, initial: np.ndarray) -> Dict[str, Any]:
    bounds = (np.zeros_like(initial), np.ones_like(initial))
    result = least_squares(simulate_confidence, x0=initial, bounds=bounds, args=(observed, e_tensor), max_nfev=5000)
    mean_obs = np.nanmean(observed, axis=0)
    pred_all = compute_predictions(result.x, observed, e_tensor)
    mean_pred = np.nanmean(pred_all, axis=0)
    mean_obs_trials = mean_obs[1:]
    mean_pred_trials = mean_pred[1:]
    ssres = np.nansum((mean_pred_trials - mean_obs_trials) ** 2)
    sstot = np.nansum((mean_obs_trials - np.nanmean(mean_obs_trials)) ** 2)
    adj_r2 = 1 - (99 / 92) * (ssres / sstot) if sstot > 0 else float("nan")
    mse_mean = ssres / (2 * Config.NUM_TRIALS)
    residuals = simulate_confidence(result.x, observed, e_tensor)
    return {"params": result.x, "mse": mse_mean, "adj_r2": adj_r2, "residuals": residuals}


def load_score_categories() -> Dict[str, List[int]]:
    categories = {"poor": [], "fair": [], "good": []}
    offsets = {1: 0, 2: 50}
    for cond in (1, 2):
        mat = loadmat(Config.CODE_DIR / f"scoregroup{cond}.mat")["group"][0]
        for cat_name, arr in zip(["poor", "fair", "good"], mat):
            ids = [int(x) for x in arr.flatten()]
            categories[cat_name].extend([offsets[cond] + i for i in ids])
    return categories


def prepare_logit(matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
    recs = []
    pid_arr = matrices["pid_array"]
    ai = matrices["ai_conf_matrix"][:, 1:]
    selfc = matrices["self_conf_matrix"][:, 1:]
    accept = matrices["act_matrix"][:, 1:]
    for i, pid in enumerate(pid_arr):
        for t in range(Config.NUM_TRIALS):
            recs.append({
                "pid": pid,
                "trial": t + 1,
                "aiconf": ai[i, t],
                "selfconf": selfc[i, t],
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
    for cat in ["poor", "fair", "good"]:
        subset = logit_df[logit_df["pid"].isin(score_groups[cat])]
        if subset.empty:
            continue
        model = smf.logit("accept ~ aiconf + selfconf + C(pid)", data=subset).fit(method="lbfgs", maxiter=1000, disp=False)
        for pred in ("aiconf", "selfconf"):
            records.append({
                "category": cat,
                "predictor": pred,
                "coef": model.params[pred],
                "std_err": model.bse[pred],
                "p_value": model.pvalues[pred],
            })
    return pd.DataFrame(records)


def run_robustness(matrices: Dict[str, np.ndarray]) -> Dict[str, Any]:
    np.random.seed(Config.RANDOM_SEED)
    n = matrices["ai_conf_matrix"].shape[0]
    subset = min(Config.ROBUSTNESS_SUBSET, n)
    res = {"aiconf": {"params": [], "mse": [], "r2": []},
           "selfconf": {"params": [], "mse": [], "r2": []}}
    for _ in range(Config.ROBUSTNESS_N_ITER):
        idx = np.random.choice(n, size=subset, replace=False)
        idx.sort()
        e_sub = matrices["e_tensor"][idx]
        for name, mat in [("aiconf", matrices["ai_conf_matrix"]), ("selfconf", matrices["self_conf_matrix"])]:
            obs = mat[idx]
            initial = np.array(Config.INITIAL[name], dtype=float)
            fit_res = fit_model(obs, e_sub, initial)
            res[name]["params"].append(fit_res["params"])
            res[name]["mse"].append(fit_res["mse"])
            res[name]["r2"].append(fit_res["adj_r2"])
    for name in res:
        res[name]["params"] = np.array(res[name]["params"])
        res[name]["mse"] = np.array(res[name]["mse"])
        res[name]["r2"] = np.array(res[name]["r2"])
    return res


def summarize_robustness(robust: Dict[str, Any], base_params: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name in ["aiconf", "selfconf"]:
        params_array = robust[name]["params"]
        mse_array = robust[name]["mse"]
        r2_array = robust[name]["r2"]
        base_row = base_params[base_params["model"] == name].iloc[0]
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
                "original": base_row[p_name],
            })
        records.append({"model": name, "parameter": "mse", "mean": mse_array.mean(), "std": mse_array.std(ddof=1), "cv": mse_array.std(ddof=1)/(mse_array.mean()+1e-10), "original": np.nan})
        records.append({"model": name, "parameter": "r2_adj", "mean": r2_array.mean(), "std": r2_array.std(ddof=1), "cv": r2_array.std(ddof=1)/(abs(r2_array.mean())+1e-10), "original": np.nan})
    return pd.DataFrame(records)


def plot_figure3(model_params: pd.DataFrame, matrices: Dict[str, np.ndarray], output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ai_params = model_params[model_params["model"] == "aiconf"].iloc[0][Config.PARAMETER_NAMES].values
    self_params = model_params[model_params["model"] == "selfconf"].iloc[0][Config.PARAMETER_NAMES].values
    cond_mask = {1: matrices["condition_array"] == 1, 2: matrices["condition_array"] == 2}
    trials = np.arange(Config.NUM_TRIALS + 1)
    for row, cond in enumerate([1, 2]):
        mask = cond_mask[cond]
        e_sub = matrices["e_tensor"][mask]

        ai_obs = matrices["ai_conf_matrix"][mask]
        ai_pred = compute_predictions(ai_params, ai_obs, e_sub)
        ai_mean = np.nanmean(ai_obs, axis=0); ai_se = np.nanstd(ai_obs, axis=0, ddof=1) / np.sqrt(ai_obs.shape[0])
        ai_pred_mean = np.nanmean(ai_pred, axis=0)
        ax_ai = axes[row, 0]
        ax_ai.errorbar(trials, ai_mean, yerr=ai_se, fmt='o', color='black', markersize=4, capsize=3, alpha=0.7, label='Observed')
        ax_ai.plot(trials, ai_pred_mean, color='#1f77b4', linewidth=2.5, label='Model')
        ax_ai.set_ylim(0, 1); ax_ai.grid(alpha=0.3)
        ax_ai.set_title(f'AI Confidence (Cond {cond})'); ax_ai.set_xlabel('Trial'); ax_ai.set_ylabel('Confidence')
        if row == 0: ax_ai.legend(loc='upper right')

        self_obs = matrices["self_conf_matrix"][mask]
        self_pred = compute_predictions(self_params, self_obs, e_sub)
        self_mean = np.nanmean(self_obs, axis=0); self_se = np.nanstd(self_obs, axis=0, ddof=1) / np.sqrt(self_obs.shape[0])
        self_pred_mean = np.nanmean(self_pred, axis=0)
        ax_self = axes[row, 1]
        ax_self.errorbar(trials, self_mean, yerr=self_se, fmt='o', color='black', markersize=4, capsize=3, alpha=0.7, label='Observed')
        ax_self.plot(trials, self_pred_mean, color='#1f77b4', linewidth=2.5, label='Model')
        ax_self.set_ylim(0, 1); ax_self.grid(alpha=0.3)
        ax_self.set_title(f'Self Confidence (Cond {cond})'); ax_self.set_xlabel('Trial'); ax_self.set_ylabel('Confidence')
        if row == 0: ax_self.legend(loc='upper right')
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
        ax.set_title(f"Condition {cond}"); ax.set_xlabel("Team Performance Score"); ax.set_ylabel("Count"); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(output_dir / "figure5_score_distributions.png", dpi=Config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_figure6(scores: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    color_map = {"poor": "#d62728", "fair": "#ff7f0e", "good": "#2ca02c"}
    for cat, sub in scores.groupby("category"):
        color = color_map.get(cat, "#1f77b4")
        axes[0].scatter(sub["team_score"], sub["individual_score"], label=cat, color=color, alpha=0.7, s=80, edgecolors="black", linewidths=0.8)
        axes[1].scatter(sub["team_score"], sub["mean_selfconf"], label=cat, color=color, alpha=0.7, s=80, edgecolors="black", linewidths=0.8)
    axes[0].set_xlabel("Team Performance"); axes[0].set_ylabel("Individual Skill"); axes[0].set_title("Team vs Individual"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Team Performance"); axes[1].set_ylabel("Avg Self-confidence"); axes[1].set_title("Team vs Self-confidence"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "figure6_scatter_plots.png", dpi=Config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    participants, trial_df = load_all_participants()
    matrices = build_matrices(participants)
    trial_df.to_csv(Config.OUTPUT_DIR / "trial_data_ration.csv", index=False)

    records = []
    for p in participants:
        main = slice(1, None)
        records.append({
            "pid": p.pid,
            "condition": p.condition,
            "acceptance_rate": np.nanmean(p.act_series[main]),
            "team_score": p.team_score,
            "individual_score": p.skill_score,
            "mean_selfconf": np.nanmean(p.self_series[main]),
            "mean_aiconf": np.nanmean(p.c_series[main]),
        })
    scores = pd.DataFrame(records)
    score_groups = load_score_categories()
    scores["category"] = "fair"
    for cat in ("poor", "good"):
        scores.loc[scores["pid"].isin(score_groups[cat]), "category"] = cat
    scores.to_csv(Config.OUTPUT_DIR / "participant_summary_ration.csv", index=False)

    records = []
    for name, mat in [("aiconf", matrices["ai_conf_matrix"]), ("selfconf", matrices["self_conf_matrix"])]:
        initial = np.array(Config.INITIAL[name], dtype=float)
        fit_res = fit_model(mat, matrices["e_tensor"], initial)
        rec = {"model": name, "mse": fit_res["mse"], "r2_adj": fit_res["adj_r2"]}
        rec.update(dict(zip(Config.PARAMETER_NAMES, fit_res["params"])))
        records.append(rec)
    model_params = pd.DataFrame(records)
    model_params.to_csv(Config.OUTPUT_DIR / "table1_model_params_ration.csv", index=False)

    plot_figure3(model_params, matrices, Config.OUTPUT_DIR)
    plot_figure4(matrices, Config.OUTPUT_DIR)
    plot_figure5(scores, score_groups, Config.OUTPUT_DIR)
    plot_figure6(scores, Config.OUTPUT_DIR)

    logit_df = prepare_logit(matrices)
    overall = run_overall_logit(logit_df); overall.to_csv(Config.OUTPUT_DIR / "table2_logit_coeffs_ration.csv", index=False)
    group_logit = run_category_logit(logit_df, score_groups); group_logit.to_csv(Config.OUTPUT_DIR / "table3_category_logit_ration.csv", index=False)

    robust = run_robustness(matrices)
    robust_stats = summarize_robustness(robust, model_params)
    robust_stats.to_csv(Config.OUTPUT_DIR / "robustness_statistics_ration.csv", index=False)
    final_params = []
    for name in ["aiconf", "selfconf"]:
        final_params.append({
            "model": name,
            "mse": robust[name]["mse"].mean(),
            "r2_adj": robust[name]["r2"].mean(),
            **dict(zip(Config.PARAMETER_NAMES, robust[name]["params"].mean(axis=0)))
        })
    pd.DataFrame(final_params).to_csv(Config.OUTPUT_DIR / "final_robust_params_ration.csv", index=False)

    print("\n=== Rational expectation model (4-term) ===")
    print(model_params.to_string(index=False))
    print("\nRobust mean params:")
    print(pd.DataFrame(final_params).to_string(index=False))


if __name__ == "__main__":
    main()

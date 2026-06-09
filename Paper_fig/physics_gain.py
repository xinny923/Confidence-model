"""
physics_gain: physics update with velocity gain and lagged memory.

Model equations (per trial n):

Evidence:
  E(n) = w1 e1(n) + w2 e2(n) + w3 e3(n) + w4 e4(n)
  (modify trials split by Config.MODIFY_AI / Config.MODIFY_SELF for +/- feedback)

Memory / expectation update:
  A(n+1) = gamma * C(n-1) + (1 - gamma) * A(n-1), gamma in [0,1]

Velocity gain:
  gain_v = 1 + ln(1 + |v(n)|)

Force (Self and AI):
  F(n) = gain_v * (alpha_fast*(E(n) - C(n))) + alpha_slow*(A(n)-C(n))

Velocity and confidence update:
  v(n+1) = F(n) + beta * v(n)
  C(n+1) = clip(C(n) + v(n+1), 0, 1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math

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
        "alpha_fast",
        "alpha_small",
        "omega1", "omega2", "omega3", "omega4",
        "beta",
        "gamma",
    ]
    INITIAL_GUESS = {
        "aiconf": [0.40, 0.30, 0.84, 0.21, 0.50, 0.00, 0.0, 0.50],
        "selfconf": [0.40, 0.30, 0.57, 0.83, 0.50, 0.00, 0.0, 0.50],
    }
    BOUNDS_LOW_DEFAULT = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    BOUNDS_HIGH_DEFAULT = np.array([2, 2, 1, 1, 1, 1, 0, 1], dtype=float)
    BETA_FIXED = 0.0

    MODIFY_AI = 1
    MODIFY_SELF = 0

    DATA_DIR = Path("Data")
    OUTPUT_DIR = Path("Paper_fig") / "physics_gain_output"
    PLOT_DPI = 300
    RANDOM_SEED = 42
    MAX_NFEV = 2000
    DELTA_WEIGHT = 0.0
    ROBUSTNESS_N_ITER = 100
    ROBUSTNESS_SUBSET_SIZE = 80
    ROBUSTNESS_ENABLED = True
    VIF_ANALYSIS = True


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
    import re
    m = re.search(r"data(\d+)_(\d+)\.csv", path.name)
    if not m:
        raise ValueError(f"Invalid filename: {path.name}")
    return int(m.group(1)), int(m.group(2))


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


def parse_participant(path: Path, pid: int, condition: int, base_id: int) -> ParticipantData:
    df = pd.read_csv(path, header=None)

    def get_num(r: int, c: int) -> float:
        return safe_float(df.iat[r, c])

    def get_str(r: int, c: int) -> str:
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
    trial_records = []

    for t in range(Config.NUM_TRIALS):
        r = Config.MAIN_START_IDX + t
        orig = get_str(r, 2)
        ai = get_str(r, 3)
        final = get_str(r, 5)
        fb = get_num(r, 7)

        if final == ai:
            action = 0
        elif final == orig:
            action = 1
        else:
            action = 2
        act_series.append(action)

        if not math.isnan(fb):
            sign = int(fb / 5 * -1)
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
            "original_move": orig,
            "ai_suggestion": ai,
            "final_move": final,
            "action": action,
            "action_label": ["accept", "reject", "modify"][action],
            "feedback": fb,
            "selfconf": self_series[t + 1],
            "aiconf": c_series[t + 1],
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
        act_series=np.array(act_series, dtype=float),
        perf_series=perf_series,
        skill_score=skill_score,
        team_score=team_score,
        trial_records=trial_records,
    )


def load_participants(data_dir: Path) -> Tuple[List[ParticipantData], pd.DataFrame]:
    file_map: Dict[int, Dict[int, Path]] = {1: {}, 2: {}}
    for path in sorted(data_dir.glob("data*_*.csv")):
        base_id, cond = parse_filename(path)
        file_map[cond][base_id] = path
    parts: List[ParticipantData] = []
    trials: List[Dict[str, Any]] = []
    pid = 1
    for cond in (1, 2):
        for base_id in sorted(file_map[cond]):
            p = parse_participant(file_map[cond][base_id], pid, cond, base_id)
            parts.append(p)
            trials.extend(p.trial_records)
            pid += 1
    return parts, pd.DataFrame(trials)


def build_matrices(parts: List[ParticipantData]) -> Dict[str, np.ndarray]:
    return {
        "ai_conf_matrix": np.vstack([p.c_series for p in parts]),
        "self_conf_matrix": np.vstack([p.self_series for p in parts]),
        "e_tensor": np.stack([p.e_matrix for p in parts]),
        "condition_array": np.array([p.condition for p in parts], dtype=int),
    }


def simulate_residuals(params: np.ndarray, observed: np.ndarray,
                       e_tensor: np.ndarray) -> np.ndarray:
    alpha_fast, alpha_slow, o1, o2, o3, o4, beta, gamma = params
    n_participants, series_len = observed.shape
    T = series_len - 1

    c = observed[:, 0].copy()
    a = observed[:, 0].copy()
    v = np.zeros_like(c)
    pred = np.zeros_like(observed)
    pred[:, 0] = c
    c_prev = c.copy()
    a_prev = a.copy()

    for t in range(T):
        base_exp = (
            o1 * e_tensor[:, t, 0] + o2 * e_tensor[:, t, 1]
            + o3 * e_tensor[:, t, 2] + o4 * e_tensor[:, t, 3]
        )
        exp_term = base_exp

        if t == 0:
            a = a_prev.copy()
        else:
            a = gamma * c_prev + (1.0 - gamma) * a_prev

        gain_v = 1.0 + np.log1p(np.abs(v))
        force = (gain_v * (alpha_fast * (exp_term - c))) + alpha_slow * (a - c)
        v = force + beta * v
        c_prev = c
        a_prev = a
        c = np.clip(c + v, 0.0, 1.0)
        pred[:, t + 1] = c

    resid = observed[:, 1:] - pred[:, 1:]
    delta_obs = observed[:, 1:] - observed[:, :-1]
    delta_pred = pred[:, 1:] - pred[:, :-1]
    delta_resid = delta_pred - delta_obs

    return np.concatenate([
        resid.T.ravel(),
        np.sqrt(Config.DELTA_WEIGHT) * delta_resid.T.ravel()
    ])


def compute_predictions(params: np.ndarray, observed: np.ndarray,
                        e_tensor: np.ndarray) -> np.ndarray:
    alpha_fast, alpha_slow, o1, o2, o3, o4, beta, gamma = params
    n_participants, series_len = observed.shape
    T = series_len - 1

    c = observed[:, 0].copy()
    a = observed[:, 0].copy()
    v = np.zeros_like(c)
    pred = np.zeros_like(observed)
    pred[:, 0] = c
    c_prev = c.copy()
    a_prev = a.copy()

    for t in range(T):
        base_exp = (
            o1 * e_tensor[:, t, 0] + o2 * e_tensor[:, t, 1]
            + o3 * e_tensor[:, t, 2] + o4 * e_tensor[:, t, 3]
        )
        exp_term = base_exp

        if t == 0:
            a = a_prev.copy()
        else:
            a = gamma * c_prev + (1.0 - gamma) * a_prev

        gain_v = 1.0 + np.log1p(np.abs(v))
        force = (gain_v * (alpha_fast * (exp_term - c))) + alpha_slow * (a - c)
        v = force + beta * v
        c_prev = c
        a_prev = a
        c = np.clip(c + v, 0.0, 1.0)
        pred[:, t + 1] = c
    return pred


def fit_model(obs: np.ndarray, e_tensor: np.ndarray,
              init: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray],
              condition_array: np.ndarray) -> Dict[str, Any]:
    res = least_squares(
        simulate_residuals,
        x0=init,
        bounds=bounds,
        args=(obs, e_tensor),
        max_nfev=Config.MAX_NFEV,
    )
    pred_all = compute_predictions(res.x, obs, e_tensor)
    params = res.x
    k = len(Config.PARAMETER_NAMES) - 1
    metrics = condition_mean_metrics(obs, pred_all, condition_array, Config.NUM_TRIALS, adjustment_k=k)
    return {"params": params, "mse": metrics["mse"], "adj_r2": metrics["adj_r2"], "r2": metrics["r2"], "pred": pred_all}


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
    n_iter = Config.ROBUSTNESS_N_ITER
    subset_size = Config.ROBUSTNESS_SUBSET_SIZE
    rng = np.random.default_rng(Config.RANDOM_SEED)
    n_participants = matrices["ai_conf_matrix"].shape[0]
    subset_size = min(subset_size, n_participants)

    results = {
        "aiconf": {"params": [], "mse": [], "r2": []},
        "selfconf": {"params": [], "mse": [], "r2": []},
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
            init = np.array(Config.INITIAL_GUESS[model_name], dtype=float)
            low = Config.BOUNDS_LOW_DEFAULT.copy()
            high = Config.BOUNDS_HIGH_DEFAULT.copy()
            beta_idx = Config.PARAMETER_NAMES.index("beta")
            init[beta_idx] = Config.BETA_FIXED
            low[beta_idx] = Config.BETA_FIXED
            high[beta_idx] = Config.BETA_FIXED + 1e-12
            fit_res = fit_model(obs_sub, e_sub, init, (low, high), cond_sub)
            results[model_name]["params"].append(fit_res["params"])
            results[model_name]["mse"].append(fit_res["mse"])
            results[model_name]["r2"].append(fit_res["r2"])

    for key in results:
        results[key]["params"] = np.array(results[key]["params"])
        results[key]["mse"] = np.array(results[key]["mse"])
        results[key]["r2"] = np.array(results[key]["r2"])
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
    parts, _ = load_participants(Config.DATA_DIR)
    matrices = build_matrices(parts)
    beta_idx = Config.PARAMETER_NAMES.index("beta")

    ai_init = np.array(Config.INITIAL_GUESS["aiconf"], dtype=float)
    ai_low = Config.BOUNDS_LOW_DEFAULT.copy()
    ai_high = Config.BOUNDS_HIGH_DEFAULT.copy()
    omega3_idx = Config.PARAMETER_NAMES.index("omega3")
    ai_init[omega3_idx] = 0.0
    ai_low[omega3_idx] = 0.0
    ai_high[omega3_idx] = 1e-12
    ai_init[beta_idx] = Config.BETA_FIXED
    ai_low[beta_idx] = Config.BETA_FIXED
    ai_high[beta_idx] = Config.BETA_FIXED + 1e-12
    ai_fit = fit_model(
        matrices["ai_conf_matrix"], matrices["e_tensor"],
        ai_init, (ai_low, ai_high), matrices["condition_array"]
    )

    self_init = np.array(Config.INITIAL_GUESS["selfconf"], dtype=float)
    self_low = Config.BOUNDS_LOW_DEFAULT.copy()
    self_high = Config.BOUNDS_HIGH_DEFAULT.copy()
    self_init[beta_idx] = Config.BETA_FIXED
    self_low[beta_idx] = Config.BETA_FIXED
    self_high[beta_idx] = Config.BETA_FIXED + 1e-12
    self_fit = fit_model(
        matrices["self_conf_matrix"], matrices["e_tensor"],
        self_init, (self_low, self_high), matrices["condition_array"]
    )

    records = []
    for name, fit in [("aiconf", ai_fit), ("selfconf", self_fit)]:
        rec = {"model": name, "mse": fit["mse"], "r2": fit["r2"]}
        rec.update(dict(zip(Config.PARAMETER_NAMES, fit["params"])))
        records.append(rec)

    df_params = pd.DataFrame(records)
    df_params.to_csv(Config.OUTPUT_DIR / "physics_gain_params.csv", index=False)

    print("\nPhysics_gain fit results:")
    print(df_params.to_string(index=False))

    plot_conf(
        ai_fit["pred"], self_fit["pred"],
        matrices["ai_conf_matrix"], matrices["self_conf_matrix"],
        matrices["condition_array"],
    )

    if Config.ROBUSTNESS_ENABLED:
        robust = run_robustness(matrices)
        robust_metrics = pd.DataFrame({
            "iteration": np.arange(1, robust["aiconf"]["mse"].shape[0] + 1),
            "aiconf_mse": robust["aiconf"]["mse"],
            "aiconf_r2": robust["aiconf"]["r2"],
            "selfconf_mse": robust["selfconf"]["mse"],
            "selfconf_r2": robust["selfconf"]["r2"],
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

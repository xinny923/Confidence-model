"""
Latent-state model with streak-modulated drift:
  - Streak S_t counts consecutive wins/losses (sign keeps track of streak direction)
  - Drift_ai: v_t = beta1*Feedback_t * (1 + delta*|S_t|)
    * Negative feedback additionally scaled by fixed k_penalty=2.25
  - S_t update:
      if fb>0: S_t = S_{t-1}+1 if prev fb>0 else 1
      if fb<0: S_t = S_{t-1}-1 if prev fb<0 else -1
      fb=0 or NaN: S_t reset to 0
  - Self state: S_state = lambda*self + beta3*user_win
  - No gap/score modulation

Outputs in output_latent/: fitted_params.csv, trajectories.npy,
figure_conf.png / figure_conf_scatter.png, figure_force.png, r2_stats.csv
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Config:
    NUM_TRIALS = 30
    PRACTICE_LAST_IDX = 4
    MAIN_START_IDX = 6
    SCORE_ROW_IDX = MAIN_START_IDX + NUM_TRIALS

    DATA_DIR = Path("Data")
    OUTPUT_DIR = Path("output_latent")

    PARAMS = [
        "lambda",
        "beta_ai",
        "beta_self",
        "k_mom_ai",
        "k_mom_self",
        "eta_ai",
        "eta_self",
        "theta_A",
        "theta_S",
    ]
    INITIAL = np.array([0.8, 0.5, 0.5, 0.3, 0.3, 0.1, 0.1, -0.35, 0.4], dtype=float)
    eps = 1e-9
    BOUNDS = (
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, -3.0], dtype=float),
        np.array([1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0], dtype=float),  # k_mom in [0,2], eta in [0,1]
    )

    GAMMA_DECISION = 1.0
    DELTA_DECISION = 0.0
    PLOT_DPI = 300


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


@dataclass
class TrialData:
    aiconf: np.ndarray
    selfconf: np.ndarray
    agreement: np.ndarray
    ai_win: np.ndarray
    user_win: np.ndarray
    feedback_norm: np.ndarray
    action_label: np.ndarray
    condition: np.ndarray


def logit(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def load_all_trials() -> TrialData:
    aiconf_list = []
    selfconf_list = []
    agree_list = []
    aiwin_list = []
    userwin_list = []
    fb_norm_list = []
    action_list = []
    cond_list = []

    for path in sorted(Config.DATA_DIR.glob("data*_*.csv")):
        df = pd.read_csv(path, header=None)
        _, condition = parse_filename(path)
        ai_conf = np.array(
            [safe_float(df.iat[Config.PRACTICE_LAST_IDX, 9])]
            + [safe_float(df.iat[Config.MAIN_START_IDX + t, 9]) for t in range(Config.NUM_TRIALS)],
            dtype=float,
        )
        self_conf = np.array(
            [safe_float(df.iat[Config.PRACTICE_LAST_IDX, 8])]
            + [safe_float(df.iat[Config.MAIN_START_IDX + t, 8]) for t in range(Config.NUM_TRIALS)],
            dtype=float,
        )
        aiconf_list.append(ai_conf)
        selfconf_list.append(self_conf)

        agree = []
        ai_win = []
        user_win = []
        actions = []
        for t in range(Config.NUM_TRIALS):
            r = Config.MAIN_START_IDX + t
            orig = safe_str(df.iat[r, 2])
            ai = safe_str(df.iat[r, 3])
            final = safe_str(df.iat[r, 5])
            fb = safe_float(df.iat[r, 7])
            agree.append(1.0 if orig == ai else 0.0)
            fb_norm = fb / 5.0 if not math.isnan(fb) else math.nan
            ai_win.append(1.0 if (final == ai and not math.isnan(fb_norm) and fb_norm > 0) else 0.0)
            user_win.append(1.0 if (final == orig and not math.isnan(fb_norm) and fb_norm > 0) else 0.0)
            if final == ai:
                actions.append("accept")
            elif final == orig:
                actions.append("reject")
            else:
                actions.append("modify")
            fb_norm_list.append(fb_norm)
        agree_list.append(np.array(agree))
        aiwin_list.append(np.array(ai_win))
        userwin_list.append(np.array(user_win))
        action_list.append(np.array(actions))
        cond_list.append(condition)

    return TrialData(
        aiconf=np.array(aiconf_list),
        selfconf=np.array(selfconf_list),
        agreement=np.array(agree_list),
        ai_win=np.array(aiwin_list),
        user_win=np.array(userwin_list),
        feedback_norm=np.array(fb_norm_list, dtype=float).reshape(len(cond_list), Config.NUM_TRIALS),
        action_label=np.array(action_list),
        condition=np.array(cond_list, dtype=int),
    )


def simulate_states(params: np.ndarray, data: TrialData) -> Dict[str, np.ndarray]:
    lam, b_ai, b_self, k_mom_ai, k_mom_self, eta_ai, eta_self, thetaA, thetaS = params
    n_participants, series_len = data.aiconf.shape
    T = series_len - 1

    A = np.zeros_like(data.aiconf)
    S = np.zeros_like(data.selfconf)
    A[:, 0] = logit(data.aiconf[:, 0])
    S[:, 0] = logit(data.selfconf[:, 0])
    A_init = A[:, 0].copy()
    V_self = np.zeros((n_participants,), dtype=float)

    for t in range(T):
        fb = data.feedback_norm[:, t]
        # Fixed attribution weights for AI / Self based on action & outcome sign (constants)
        fb_ai = np.zeros_like(fb)
        fb_self = np.zeros_like(fb)
        for i in range(n_participants):
            if math.isnan(fb[i]) or fb[i] == 0:
                continue
            act = data.action_label[i, t]
            sign = 1.0 if fb[i] > 0 else -1.0
            # AI weights: accept=1.0, modify=0.5, reject=0.2
            if act == "accept":
                fb_ai[i] = sign * 1.0
            elif act == "modify":
                fb_ai[i] = sign * 0.5
            elif act == "reject":
                fb_ai[i] = sign * 0.2
            # Self weights: reject=1.0, accept=0.0, modify=0.5
            if act == "reject":
                fb_self[i] = sign * 1.0
            elif act == "accept":
                fb_self[i] = 0.0
            elif act == "modify":
                fb_self[i] = sign * 0.5
            # Alignment bonus: if orig == ai, give both sides the same signed feedback
            if data.agreement[i, t] == 1:
                fb_ai[i] = sign * 1.0

        fb_ai_clean = np.where(np.isnan(fb_ai), 0.0, fb_ai)
        fb_self_clean = np.where(np.isnan(fb_self), 0.0, fb_self)
        evidence_ai = b_ai * fb_ai_clean
        evidence_self = b_self * fb_self_clean

        V_self = (1 - eta_self) * V_self + k_mom_self * evidence_self

        A_next = lam * A[:, t] + evidence_ai  # AI side: no momentum term
        A[:, t + 1] = A_next  # no AI anchor

        self_next = lam * S[:, t] + evidence_self + V_self
        S[:, t + 1] = (1 - 0.0) * self_next + 0.0  # no self anchor

    C_ai_hat = sigmoid(A - thetaA)
    C_self_hat = sigmoid(S - thetaS)
    return {"A": A, "S": S, "C_ai_hat": C_ai_hat, "C_self_hat": C_self_hat}


def residuals(params: np.ndarray, data: TrialData) -> np.ndarray:
    sim = simulate_states(params, data)
    ai_obs = data.aiconf[:, 1:]
    self_obs = data.selfconf[:, 1:]
    ai_pred = sim["C_ai_hat"][:, 1:]
    self_pred = sim["C_self_hat"][:, 1:]
    res = np.concatenate([ai_obs - ai_pred, self_obs - self_pred], axis=1)
    return res.ravel()


def fit_model(data: TrialData) -> Tuple[np.ndarray, float]:
    result = least_squares(
        residuals,
        x0=Config.INITIAL,
        bounds=Config.BOUNDS,
        args=(data,),
        max_nfev=5000,
    )
    mse = np.mean(residuals(result.x, data) ** 2)
    return result.x, mse


def compute_decision_force(sim: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    D = sim["A"] - sim["S"]
    gamma = Config.GAMMA_DECISION
    delta = Config.DELTA_DECISION
    P_reject = 1 / (1 + np.exp(gamma * (D + delta)))
    return {"D": D, "P_reject": P_reject}


def plot_confidence(sim: Dict[str, np.ndarray], data: TrialData, output_dir: Path):
    trials = np.arange(1, Config.NUM_TRIALS + 1)
    change = 20
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    conds = [1, 2]
    titles = {(0, 0): "AI confidence (Cond1 high→low)", (0, 1): "AI confidence (Cond2 low→high)",
              (1, 0): "Self confidence (Cond1)", (1, 1): "Self confidence (Cond2)"}
    for col, cond in enumerate(conds):
        idx = np.where(data.condition == cond)[0]
        if len(idx) == 0:
            continue
        for row, (obs_mat, pred_mat) in enumerate([(data.aiconf, sim["C_ai_hat"]), (data.selfconf, sim["C_self_hat"])]):
            obs = obs_mat[idx][:, 1:]
            pred = pred_mat[idx][:, 1:]
            mean_obs = np.nanmean(obs, axis=0)
            se_obs = np.nanstd(obs, axis=0, ddof=1) / np.sqrt(obs.shape[0])
            mean_pred = np.nanmean(pred, axis=0)
            ax = axes[row, col]
            if cond == 1:
                ax.axvspan(1, change, color="#9ecae1", alpha=0.25)
                ax.axvspan(change, Config.NUM_TRIALS, color="#fee6ce", alpha=0.25)
            else:
                ax.axvspan(1, change, color="#fee6ce", alpha=0.25)
                ax.axvspan(change, Config.NUM_TRIALS, color="#9ecae1", alpha=0.25)
            ax.axvline(change, color="orange", linestyle="--", linewidth=2, alpha=0.8)
            ax.errorbar(trials, mean_obs, yerr=se_obs, fmt="o", color="black", markersize=4, capsize=2, label="Data")
            ax.plot(trials, mean_pred, color="#1f77b4", linewidth=2.5, label="Average model fit")
            ax.set_ylim(0, 1)
            ax.set_xlim(1, Config.NUM_TRIALS)
            ax.grid(alpha=0.3)
            ax.set_title(titles[(row, col)])
            if row == 1:
                ax.set_xlabel("Trial")
            ax.set_ylabel("Confidence")
            if row == 0 and col == 0:
                ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure_conf.png", dpi=Config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, obs, pred, title in [
        (axes[0], data.aiconf[:, 1:].ravel(), sim["C_ai_hat"][:, 1:].ravel(), "AI confidence"),
        (axes[1], data.selfconf[:, 1:].ravel(), sim["C_self_hat"][:, 1:].ravel(), "Self confidence"),
    ]:
        ax.scatter(obs, pred, alpha=0.3, s=10, color="#2ca02c", edgecolors="none")
        ax.plot([0, 1], [0, 1], color="red", linestyle="--")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{title}: scatter")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_conf_scatter.png", dpi=Config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_force(sim: Dict[str, np.ndarray], data: TrialData, output_dir: Path):
    force = compute_decision_force(sim)
    D = force["D"][:, 1:]
    P = force["P_reject"][:, 1:]
    actions = data.action_label
    recs = []
    for i in range(actions.shape[0]):
        for t in range(Config.NUM_TRIALS):
            recs.append({"action": actions[i, t], "D": D[i, t], "P_reject": P[i, t]})
    df = pd.DataFrame(recs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = ["accept", "modify", "reject"]
    colors = {"accept": "#2ca02c", "modify": "#ff7f0e", "reject": "#d62728"}
    data_for_box = [df[df["action"] == a]["D"].dropna() for a in order]
    bp = axes[0].boxplot(data_for_box, tick_labels=order, patch_artist=True, showfliers=False)
    for patch, a in zip(bp["boxes"], order):
        patch.set_facecolor(colors[a]); patch.set_alpha(0.6)
    axes[0].set_title("Decision force D = A - S by action")
    axes[0].set_ylabel("D")
    axes[0].grid(alpha=0.3)

    ax = axes[1]
    markers = {"accept": "o", "modify": "^", "reject": "s"}
    for a in order:
        sub = df[df["action"] == a]
        if len(sub) > 1500:
            sub = sub.sample(1500, random_state=42)
        ax.scatter(sub["D"], sub["P_reject"], s=25, alpha=0.45, label=a, color=colors[a],
                   edgecolors="black", linewidths=0.3, marker=markers.get(a, "o"))
    dmin, dmax = df["D"].min(), df["D"].max()
    d_lin = np.linspace(dmin, dmax, 300)
    P_curve = 1 / (1 + np.exp(Config.GAMMA_DECISION * (d_lin + Config.DELTA_DECISION)))
    ax.plot(d_lin, P_curve, color="black", linewidth=2, label="Sigmoid(D)")
    ax.set_xlabel("D = A - S")
    ax.set_ylabel("P(reject)")
    ax.set_title("P(reject) vs decision force")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "figure_force.png", dpi=Config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    stats = df.groupby("action").agg(
        D_mean=("D", "mean"),
        D_std=("D", "std"),
        P_mean=("P_reject", "mean"),
        P_std=("P_reject", "std"),
        count=("D", "size"),
    )
    stats.to_csv(output_dir / "force_stats_by_action.csv")


def main():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all_trials()
    params, mse = fit_model(data)
    sim = simulate_states(params, data)

    pd.DataFrame([dict(zip(Config.PARAMS, params)) | {"mse": mse}]).to_csv(
        Config.OUTPUT_DIR / "fitted_params.csv", index=False
    )
    np.save(Config.OUTPUT_DIR / "trajectories.npy", sim)

    plot_confidence(sim, data, Config.OUTPUT_DIR)
    plot_force(sim, data, Config.OUTPUT_DIR)

    def r2_stats(obs: np.ndarray, pred: np.ndarray, k: int) -> Tuple[float, float]:
        mask = ~np.isnan(obs)
        obs = obs[mask]; pred = pred[mask]
        ssres = np.sum((pred - obs) ** 2)
        sstot = np.sum((obs - np.mean(obs)) ** 2)
        r2 = 1 - ssres / sstot if sstot > 0 else float("nan")
        n = len(obs)
        adj = 1 - (ssres / (n - k - 1)) / (sstot / (n - 1)) if n > k + 1 and sstot > 0 else float("nan")
        return r2, adj

    ai_r2_pt, ai_adj_pt = r2_stats(data.aiconf[:, 1:].ravel(), sim["C_ai_hat"][:, 1:].ravel(), k=len(Config.PARAMS))
    self_r2_pt, self_adj_pt = r2_stats(data.selfconf[:, 1:].ravel(), sim["C_self_hat"][:, 1:].ravel(), k=len(Config.PARAMS))

    def r2_agg(obs_mat: np.ndarray, pred_mat: np.ndarray) -> float:
        mean_obs = np.nanmean(obs_mat[:, 1:], axis=0)
        mean_pred = np.nanmean(pred_mat[:, 1:], axis=0)
        ssres = np.sum((mean_pred - mean_obs) ** 2)
        sstot = np.sum((mean_obs - np.mean(mean_obs)) ** 2)
        return 1 - ssres / sstot if sstot > 0 else float("nan")

    ai_r2_agg = r2_agg(data.aiconf, sim["C_ai_hat"])
    self_r2_agg = r2_agg(data.selfconf, sim["C_self_hat"])

    pd.DataFrame([
        {"target": "ai", "r2_point": ai_r2_pt, "adj_r2_point": ai_adj_pt, "r2_agg": ai_r2_agg},
        {"target": "self", "r2_point": self_r2_pt, "adj_r2_point": self_adj_pt, "r2_agg": self_r2_agg},
    ]).to_csv(Config.OUTPUT_DIR / "r2_stats.csv", index=False)

    logger.info("Fitted params: %s", dict(zip(Config.PARAMS, params)))
    logger.info("MSE: %.6f", mse)
    logger.info("Pointwise R2 AI: %.4f adj: %.4f | Self: %.4f adj: %.4f", ai_r2_pt, ai_adj_pt, self_r2_pt, self_adj_pt)
    logger.info("Aggregated R2 (mean per trial) AI: %.4f | Self: %.4f", ai_r2_agg, self_r2_agg)


if __name__ == "__main__":
    main()

"""
AI-credit weighting model with four scenarios (S1–S4) and no omega terms.

Scenarios (per trial):
  S1 (Follow-AI):    orig != ai, final == ai
  S2 (Align-All):    orig == ai == final
  S3 (Reject-AI):    orig != ai, final == orig
  S4 (Modify/Other): otherwise

Experience term E_t = c_scenario (constant credit weight). Confidence update
follows the original 3-alpha + gamma dynamics. Parameters:
  alpha_e, alpha_a, alpha_b, c1, c2, c3, c4, gamma

Outputs:
  - Model parameters (AI/Self) with adj R^2 (99/92 factor) and MSE (mean curve)
  - Robustness (100 x 80 participants) mean parameters
  - Figure 3: predicted vs observed curves and scatter
  - Figure 4: confidence trends by condition
  Saved to output_teeth/
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
        "c1_pos", "c1_neg",
        "c2",
        "c3_pos", "c3_neg",
        "c4",
        "gamma",
    ]

    INITIAL = {
        "aiconf": [0.27, 0.34, 0.05, 0.65, 0.45, 0.70, 0.40, 0.30, 0.50, 0.35],
        "selfconf": [0.28, 0.47, 0.00, 0.60, 0.40, 0.70, 0.35, 0.25, 0.45, 0.12],
    }

    OUTPUT_DIR = Path("output_teeth")
    DATA_DIR = Path("Data")
    CODE_DIR = Path("Code")
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
    credit_series: np.ndarray  # (30,) placeholder (actual credit from params)
    scenario_series: np.ndarray  # (30,) int 1..4
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


def classify_scenario(orig: str, ai: str, final: str) -> int:
    if orig != ai and final == ai:
        return 1  # Follow-AI
    if orig == ai == final:
        return 2  # Align-All
    if orig != ai and final == orig:
        return 3  # Reject-AI
    return 4  # Modify / other


def parse_participant(path: Path, pid: int, condition: int, base_id: int) -> ParticipantData:
    df = pd.read_csv(path, header=None)

    def get_num(r, c):
        return safe_float(df.iat[r, c])

    def get_str(r, c):
        return safe_str(df.iat[r, c])

    c_series = np.array(
        [get_num(Config.PRACTICE_LAST_IDX, 9)]
        + [get_num(Config.MAIN_START_IDX + t, 9) for t in range(Config.NUM_TRIALS)],
        dtype=float,
    )
    self_series = np.array(
        [get_num(Config.PRACTICE_LAST_IDX, 8)]
        + [get_num(Config.MAIN_START_IDX + t, 8) for t in range(Config.NUM_TRIALS)],
        dtype=float,
    )

    credit_series = np.ones(Config.NUM_TRIALS, dtype=float)
    scenario_series = np.zeros(Config.NUM_TRIALS, dtype=int)
    act_series = [0]
    trial_records: List[Dict[str, Any]] = []

    for t in range(Config.NUM_TRIALS):
        r = Config.MAIN_START_IDX + t
        orig = get_str(r, 2)
        ai = get_str(r, 3)
        final = get_str(r, 5)
        feedback2 = get_num(r, 7)

        scenario = classify_scenario(orig, ai, final)
        scenario_series[t] = scenario

        action_label = "accept" if final == ai else ("reject" if final == orig else "modify")
        accept = 1 if final == ai else 0
        act_series.append(accept)

        trial_records.append(
            {
                "pid": pid,
                "condition": condition,
                "trial": t + 1,
                "action_label": action_label,
                "scenario": scenario,
                "feedback2": feedback2,
                "selfconf": get_num(r, 8),
                "aiconf": get_num(r, 9),
            }
        )

    perf_series = np.array(
        [get_num(Config.PRACTICE_LAST_IDX, 7)]
        + [get_num(Config.MAIN_START_IDX + t, 7) for t in range(Config.NUM_TRIALS)],
        dtype=float,
    )
    skill_score = np.nansum([get_num(Config.MAIN_START_IDX + t, 6) for t in range(Config.NUM_TRIALS)])
    team_score = get_num(Config.SCORE_ROW_IDX, 1)

    return ParticipantData(
        pid=pid,
        base_id=base_id,
        condition=condition,
        c_series=c_series,
        self_series=self_series,
        credit_series=credit_series,
        scenario_series=scenario_series,
        act_series=np.array(act_series, dtype=float),
        perf_series=perf_series,
        skill_score=skill_score,
        team_score=team_score,
        trial_records=trial_records,
    )


def load_all_participants() -> Tuple[List[ParticipantData], pd.DataFrame]:
    logger.info("Loading participants...")
    file_map: Dict[int, Dict[int, Path]] = {1: {}, 2: {}}
    for p in sorted(Config.DATA_DIR.glob("data*_*.csv")):
        bid, cond = parse_filename(p)
        file_map[cond][bid] = p

    participants: List[ParticipantData] = []
    trial_records: List[Dict[str, Any]] = []
    pid = 1
    for cond in (1, 2):
        for bid in sorted(file_map[cond]):
            part = parse_participant(file_map[cond][bid], pid, cond, bid)
            participants.append(part)
            trial_records.extend(part.trial_records)
            pid += 1

    return participants, pd.DataFrame(trial_records)


def build_matrices(participants: List[ParticipantData]) -> Dict[str, np.ndarray]:
    ai = np.vstack([p.c_series for p in participants])
    selfc = np.vstack([p.self_series for p in participants])
    scen = np.vstack([p.scenario_series for p in participants]).astype(int)
    credit = np.vstack([p.credit_series for p in participants])
    feedback_rows = []
    for p in participants:
        row = []
        for rec in p.trial_records:
            fb = rec["feedback2"]
            row.append(fb / 5.0 if not math.isnan(fb) else math.nan)
        feedback_rows.append(row)
    feedback = np.array(feedback_rows, dtype=float)
    return {
        "ai_conf_matrix": ai,
        "self_conf_matrix": selfc,
        "aiconf_matrix": ai,
        "selfconf_matrix": selfc,
        "credit_matrix": credit,
        "scenario_matrix": scen,
        "feedback_matrix": feedback,
    }


def simulate_confidence(
    params: np.ndarray,
    observed: np.ndarray,
    scenario_matrix: np.ndarray,
    feedback_matrix: np.ndarray,
) -> np.ndarray:
    """
    Experience = credit weight by scenario + feedback sign:
      S1: c1_pos / c1_neg
      S2: c2
      S3: c3_pos / c3_neg
      S4: c4
    """
    alpha_e, alpha_a, alpha_b, c1p, c1n, c2, c3p, c3n, c4, gamma = params
    credits_base = np.array([c1p, c1n, c2, c3p, c3n, c4], dtype=float)

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
        scen = scenario_matrix[:, t]
        experience = np.zeros_like(c)

        fb = feedback_matrix[:, t]
        mask = ~np.isnan(fb)
        sign_pos = fb >= 0

        # S1
        m1 = mask & (scen == 1)
        experience[m1] = np.where(sign_pos[m1], credits_base[0], credits_base[1])
        # S2
        m2 = mask & (scen == 2)
        experience[m2] = credits_base[2]
        # S3
        m3 = mask & (scen == 3)
        experience[m3] = np.where(sign_pos[m3], credits_base[3], credits_base[4])
        # S4
        m4 = mask & (scen == 4)
        experience[m4] = credits_base[5]

        c = c + alpha_e * (experience - c) + alpha_a * (a - c) + alpha_b * (b - c)
        predicted[:, t + 1] = c

    residuals = observed[:, 1:] - predicted[:, 1:]
    return residuals.T.ravel()


def compute_predictions(
    params: np.ndarray,
    observed: np.ndarray,
    scenario_matrix: np.ndarray,
    feedback_matrix: np.ndarray,
) -> np.ndarray:
    alpha_e, alpha_a, alpha_b, c1p, c1n, c2, c3p, c3n, c4, gamma = params
    credits_base = np.array([c1p, c1n, c2, c3p, c3n, c4], dtype=float)
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
        scen = scenario_matrix[:, t]
        fb = feedback_matrix[:, t]
        experience = np.zeros_like(c)
        mask = ~np.isnan(fb)
        sign_pos = fb >= 0
        m1 = mask & (scen == 1)
        experience[m1] = np.where(sign_pos[m1], credits_base[0], credits_base[1])
        m2 = mask & (scen == 2)
        experience[m2] = credits_base[2]
        m3 = mask & (scen == 3)
        experience[m3] = np.where(sign_pos[m3], credits_base[3], credits_base[4])
        m4 = mask & (scen == 4)
        experience[m4] = credits_base[5]
        c = c + alpha_e * (experience - c) + alpha_a * (a - c) + alpha_b * (b - c)
        predicted[:, t + 1] = c
    return predicted


def fit_model(
    observed: np.ndarray,
    scenario_matrix: np.ndarray,
    feedback_matrix: np.ndarray,
    initial: np.ndarray,
) -> Dict[str, Any]:
    bounds = (np.zeros_like(initial), np.ones_like(initial))
    result = least_squares(
        simulate_confidence,
        x0=initial,
        bounds=bounds,
        args=(observed, scenario_matrix, feedback_matrix),
        max_nfev=5000,
    )
    residuals = simulate_confidence(result.x, observed, scenario_matrix, feedback_matrix)
    sse = np.sum(residuals**2)
    pred = compute_predictions(result.x, observed, scenario_matrix, feedback_matrix)
    mean_obs = np.nanmean(observed, axis=0)
    mean_pred = np.nanmean(pred, axis=0)
    mean_obs_trials = mean_obs[1:]
    mean_pred_trials = mean_pred[1:]
    ssres = np.nansum((mean_pred_trials - mean_obs_trials) ** 2)
    sstot = np.nansum((mean_obs_trials - np.nanmean(mean_obs_trials)) ** 2)
    k = len(Config.PARAMETER_NAMES) - 1  # predictors count (exclude intercept analog)
    adj_factor = 99 / (99 - k) if (99 - k) > 0 else math.nan
    adj_r2 = 1 - adj_factor * (ssres / sstot) if sstot > 0 else float("nan")
    mse_mean = ssres / (2 * Config.NUM_TRIALS)
    return {"params": result.x, "mse": mse_mean, "adj_r2": adj_r2}


def run_robustness(matrices: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    np.random.seed(Config.RANDOM_SEED)
    n_participants = matrices["ai_conf_matrix"].shape[0]
    subset_size = min(Config.ROBUSTNESS_SUBSET, n_participants)
    results: Dict[str, Dict[str, List[np.ndarray]]] = {
        "aiconf": {"params": [], "mse": [], "r2": []},
        "selfconf": {"params": [], "mse": [], "r2": []},
    }
    for _ in range(Config.ROBUSTNESS_N_ITER):
        idx = np.random.choice(n_participants, size=subset_size, replace=False)
        idx = np.sort(idx)
        scen = matrices["scenario_matrix"][idx]
        fb = matrices["feedback_matrix"][idx]
        for name, mat in [("aiconf", matrices["ai_conf_matrix"][idx]), ("selfconf", matrices["self_conf_matrix"][idx])]:
            initial = np.array(Config.INITIAL[name], dtype=float)
            fit_res = fit_model(mat, scen, fb, initial)
            results[name]["params"].append(fit_res["params"])
            results[name]["mse"].append(fit_res["mse"])
            results[name]["r2"].append(fit_res["adj_r2"])
    for name in results:
        for key in results[name]:
            results[name][key] = np.array(results[name][key])
    return results


def plot_figure3(model_params: pd.DataFrame, matrices: Dict[str, np.ndarray], output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    trials = np.arange(0, Config.NUM_TRIALS + 1)
    for row, (name, title) in enumerate([("aiconf", "AI Confidence"), ("selfconf", "Self Confidence")]):
        params = model_params[model_params["model"] == name][Config.PARAMETER_NAMES].values[0]
        pred = compute_predictions(params, matrices[f"{name}_matrix"], matrices["scenario_matrix"], matrices["feedback_matrix"])
        obs = matrices[f"{name}_matrix"]
        mean_obs = np.nanmean(obs, axis=0)
        se_obs = np.nanstd(obs, axis=0, ddof=1) / np.sqrt(obs.shape[0])
        mean_pred = np.nanmean(pred, axis=0)

        ax_mean = axes[row, 0]
        ax_mean.fill_between(trials, mean_obs - se_obs, mean_obs + se_obs, color="#e0e0e0", alpha=0.6, label="Obs ± SE")
        ax_mean.plot(trials, mean_obs, color="#000", linewidth=2, label="Observed")
        ax_mean.plot(trials, mean_pred, color="#1f77b4", linewidth=2.5, label="Model")
        ax_mean.set_ylim(0, 1)
        ax_mean.set_xlabel("Trial")
        ax_mean.set_ylabel(title)
        ax_mean.set_title(f"{title}: Mean vs Model")
        ax_mean.legend()
        ax_mean.grid(alpha=0.3)

        ax_ind = axes[row, 1]
        ax_ind.scatter(obs[:, 1:].ravel(), pred[:, 1:].ravel(), alpha=0.3, s=10, color="#2ca02c", edgecolors="none")
        ax_ind.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.5)
        ax_ind.set_xlim(0, 1)
        ax_ind.set_ylim(0, 1)
        ax_ind.set_xlabel("Observed")
        ax_ind.set_ylabel("Predicted")
        ax_ind.set_title(f"{title}: Pred vs Obs (all points)")
        ax_ind.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "figure3_model_fit.png", dpi=Config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_trends(matrices: Dict[str, np.ndarray], output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    trials = np.arange(1, Config.NUM_TRIALS + 1)
    change_point = Config.PERF_CHANGE_TRIAL
    for row, name in enumerate(["ai_conf_matrix", "self_conf_matrix"]):
        for col, (cond, data) in enumerate({1: matrices[name][:50], 2: matrices[name][50:]} .items()):
            mean = np.nanmean(data[:, 1:], axis=0)
            se = np.nanstd(data[:, 1:], axis=0, ddof=1) / np.sqrt(data.shape[0])
            ax = axes[row, col]
            ax.errorbar(trials, mean, yerr=se, fmt="o", color="black", markersize=4, capsize=2, alpha=0.6)
            t1 = trials[:change_point]
            t2 = trials[change_point:]
            if len(t1) > 1:
                coef1 = np.polyfit(t1, mean[:change_point], 1)
                ax.plot(t1, np.poly1d(coef1)(t1), color="#1f77b4", linewidth=2.5)
            if len(t2) > 1:
                coef2 = np.polyfit(t2, mean[change_point:], 1)
                ax.plot(t2, np.poly1d(coef2)(t2), color="#1f77b4", linewidth=2.5)
            ax.axvline(change_point, color="orange", linestyle="--", linewidth=2, alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_title(f"Condition {cond} ({'AI' if row==0 else 'Self'})")
            ax.set_xlabel("Trial")
            ax.set_ylabel("Confidence")
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "figure4_confidence_trends.png", dpi=Config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def prepare_logit(matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
    ai = matrices["ai_conf_matrix"][:, 1:]
    selfc = matrices["self_conf_matrix"][:, 1:]
    records = []
    for pid in range(ai.shape[0]):
        for t in range(ai.shape[1]):
            records.append({"pid": pid + 1, "trial": t + 1, "aiconf": ai[pid, t], "selfconf": selfc[pid, t]})
    return pd.DataFrame(records)


def run_overall_logit(df: pd.DataFrame) -> pd.DataFrame:
    model = smf.logit("y ~ aiconf + selfconf", data=df.assign(y=(df["trial"] > 15).astype(int)))
    res = model.fit(disp=False)
    out = res.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    return out[["term", "Coef.", "Std.Err.", "z", "P>|z|"]]


def main():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    participants, trial_df = load_all_participants()
    matrices = build_matrices(participants)
    trial_df.to_csv(Config.OUTPUT_DIR / "trial_data_teeth.csv", index=False)

    records = []
    for p in participants:
        records.append(
            {
                "pid": p.pid,
                "condition": p.condition,
                "team_score": p.team_score,
                "individual_score": p.skill_score,
                "mean_selfconf": np.nanmean(p.self_series[1:]),
                "mean_aiconf": np.nanmean(p.c_series[1:]),
            }
        )
    scores = pd.DataFrame(records)
    scores.to_csv(Config.OUTPUT_DIR / "participant_summary_teeth.csv", index=False)

    # Fit models
    rows = []
    for name, mat in [("aiconf", matrices["ai_conf_matrix"]), ("selfconf", matrices["self_conf_matrix"])]:
        initial = np.array(Config.INITIAL[name], dtype=float)
        fit_res = fit_model(mat, matrices["scenario_matrix"], matrices["feedback_matrix"], initial)
        row = {"model": name, "mse": fit_res["mse"], "r2_adj": fit_res["adj_r2"]}
        row.update(dict(zip(Config.PARAMETER_NAMES, fit_res["params"])))
        rows.append(row)
    model_params = pd.DataFrame(rows)
    model_params.to_csv(Config.OUTPUT_DIR / "table1_model_params_teeth.csv", index=False)

    # Figures
    plot_figure3(model_params, matrices, Config.OUTPUT_DIR)
    plot_confidence_trends(matrices, Config.OUTPUT_DIR)

    # Robustness
    robust = run_robustness(matrices)
    robust_rows = []
    for name in ["aiconf", "selfconf"]:
        robust_rows.append(
            {
                "model": name,
                "mse": robust[name]["mse"].mean(),
                "r2_adj": robust[name]["r2"].mean(),
                **dict(zip(Config.PARAMETER_NAMES, robust[name]["params"].mean(axis=0))),
            }
        )
    pd.DataFrame(robust_rows).to_csv(Config.OUTPUT_DIR / "final_robust_params_teeth.csv", index=False)

    print("\n=== AI-credit (S1–S4) model ===")
    print(model_params.to_string(index=False))
    print("\nRobust mean params:")
    print(pd.DataFrame(robust_rows).to_string(index=False))


if __name__ == "__main__":
    main()

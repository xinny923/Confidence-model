"""
Shared utilities for original 2-action (accept/reject), 4-experience model.
Uses output_benchmark/table1_model_params.csv produced by benchmark.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math
import re
import difflib

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)


class ConfigOrig:
    NUM_TRIALS = 30
    PRACTICE_LAST_IDX = 4
    MAIN_START_IDX = 6
    SCORE_ROW_IDX = MAIN_START_IDX + NUM_TRIALS

    PARAMETER_NAMES = [
        "alpha_e", "alpha_a", "alpha_b",
        "omega1", "omega2", "omega3", "omega4",
        "gamma",
    ]

    DATA_DIR = Path("Data")
    OUTPUT_DIR = Path("output_benchmark")
    PARAMS_PATH = Path("output_benchmark") / "table1_model_params.csv"  # from benchmark.py
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
    """1 - max similarity to AI or original; 0 if identical to either."""
    if final_move == ai_suggestion or final_move == original_move:
        return 0.0
    sim_ai = difflib.SequenceMatcher(None, str(final_move), str(ai_suggestion)).ratio() if ai_suggestion else 0.0
    sim_orig = difflib.SequenceMatcher(None, str(final_move), str(original_move)).ratio() if original_move else 0.0
    return 1.0 - max(sim_ai, sim_orig)


def bootstrap_ci(data: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Basic bootstrap CI for the mean."""
    arr = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.shape[0]), replace=True).mean(axis=1)
    lower, upper = np.percentile(samples, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(lower), float(upper)


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


def parse_participant_data(path: Path, pid: int, condition: int, base_id: int) -> ParticipantData:
    df = pd.read_csv(path, header=None)

    def get_num(row: int, col: int) -> float:
        return safe_float(df.iat[row, col])

    def get_str(row: int, col: int) -> str:
        return safe_str(df.iat[row, col])

    c_series = np.array(
        [get_num(ConfigOrig.PRACTICE_LAST_IDX, 9)] +
        [get_num(ConfigOrig.MAIN_START_IDX + t, 9) for t in range(ConfigOrig.NUM_TRIALS)],
        dtype=float
    )
    self_series = np.array(
        [get_num(ConfigOrig.PRACTICE_LAST_IDX, 8)] +
        [get_num(ConfigOrig.MAIN_START_IDX + t, 8) for t in range(ConfigOrig.NUM_TRIALS)],
        dtype=float
    )

    e_matrix = np.zeros((ConfigOrig.NUM_TRIALS, 4), dtype=float)
    act_values = [0]
    trial_records: List[Dict[str, Any]] = []

    for t in range(ConfigOrig.NUM_TRIALS):
        row_idx = ConfigOrig.MAIN_START_IDX + t
        original_move = get_str(row_idx, 2)
        ai_suggestion = get_str(row_idx, 3)
        final_move = get_str(row_idx, 5)
        feedback2 = get_num(row_idx, 7)

        if final_move == ai_suggestion:
            accept = 1
            action_label = "accept"
        elif final_move == original_move:
            accept = 0
            action_label = "reject"
        else:
            accept = 0
            action_label = "modify"
        act_values.append(accept)

        if not math.isnan(feedback2):
            sign = int(feedback2 / 5 * -1)
            idx = (sign + 1) if accept else (sign + 2)
            if 0 <= idx < 4:
                e_matrix[t, idx] = 1.0

        trial_records.append({
            "pid": pid,
            "condition": condition,
            "trial": t + 1,
            "original_move": original_move,
            "ai_suggestion": ai_suggestion,
            "final_move": final_move,
            "accept": accept,
            "action_label": action_label,
            "modify_magnitude": compute_modification_magnitude(original_move, ai_suggestion, final_move),
            "orig_eq_ai": int(original_move == ai_suggestion),
            "feedback1": get_num(row_idx, 6),
            "feedback2": feedback2,
            "selfconf": get_num(row_idx, 8),
            "aiconf": get_num(row_idx, 9),
        })

    perf_series = np.array(
        [get_num(ConfigOrig.PRACTICE_LAST_IDX, 7)] +
        [get_num(ConfigOrig.MAIN_START_IDX + t, 7) for t in range(ConfigOrig.NUM_TRIALS)],
        dtype=float
    )
    skill_score = np.nansum([get_num(ConfigOrig.MAIN_START_IDX + t, 6) for t in range(ConfigOrig.NUM_TRIALS)])
    team_score = get_num(ConfigOrig.SCORE_ROW_IDX, 1)

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
        "act_matrix": np.vstack([p.act_series for p in participants]),
        "pid_array": np.array([p.pid for p in participants], dtype=int),
        "condition_array": np.array([p.condition for p in participants], dtype=int),
    }


def simulate_confidence_dynamics(params: np.ndarray, observed: np.ndarray, e_tensor: np.ndarray) -> np.ndarray:
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


def compute_model_predictions(params: np.ndarray, observed: np.ndarray, e_tensor: np.ndarray) -> np.ndarray:
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


def fit_confidence_model(observed: np.ndarray, e_tensor: np.ndarray, initial_params: np.ndarray) -> Dict[str, Any]:
    bounds = (np.zeros_like(initial_params), np.ones_like(initial_params))
    result = least_squares(
        simulate_confidence_dynamics,
        x0=initial_params,
        bounds=bounds,
        args=(observed, e_tensor),
        max_nfev=5000,
    )
    residuals = simulate_confidence_dynamics(result.x, observed, e_tensor)
    n_obs = residuals.size
    sse = np.sum(residuals ** 2)
    observed_flat = observed[:, 1:].ravel()
    sst = np.nansum((observed_flat - np.nanmean(observed_flat)) ** 2)
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    n_params = len(result.x)
    adj_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - n_params - 1) if n_obs > n_params + 1 else float("nan")
    return {"params": result.x, "mse": sse / n_obs, "adj_r2": adj_r2}


def load_model_params(path: Path = None) -> pd.DataFrame:
    path = path or ConfigOrig.PARAMS_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run analyze_refined.py first to produce table1_model_params.csv.")
    return pd.read_csv(path)


def compute_error_dataframe(matrices: Dict[str, np.ndarray], trial_df: pd.DataFrame, model_params: pd.DataFrame) -> pd.DataFrame:
    ai_params = model_params[model_params["model"] == "aiconf"].iloc[0][ConfigOrig.PARAMETER_NAMES].values
    self_params = model_params[model_params["model"] == "selfconf"].iloc[0][ConfigOrig.PARAMETER_NAMES].values

    ai_pred = compute_model_predictions(ai_params, matrices["ai_conf_matrix"], matrices["e_tensor"])
    self_pred = compute_model_predictions(self_params, matrices["self_conf_matrix"], matrices["e_tensor"])

    ai_errors = matrices["ai_conf_matrix"][:, 1:] - ai_pred[:, 1:]
    self_errors = matrices["self_conf_matrix"][:, 1:] - self_pred[:, 1:]

    error_records = []
    for idx, pid in enumerate(matrices["pid_array"]):
        for trial in range(ConfigOrig.NUM_TRIALS):
            tr = trial_df[(trial_df["pid"] == pid) & (trial_df["trial"] == trial + 1)]
            if tr.empty:
                continue
            row = tr.iloc[0]
            error_records.append({
                "pid": pid,
                "trial": trial + 1,
                "action_label": row["action_label"],  # accept/reject
                "modify_magnitude": row.get("modify_magnitude", 0.0),
                "orig_eq_ai": int(row.get("orig_eq_ai", 0)),
                "ai_error": ai_errors[idx, trial],
                "self_error": self_errors[idx, trial],
                "ai_error_abs": abs(ai_errors[idx, trial]),
                "self_error_abs": abs(self_errors[idx, trial]),
            })
    error_df = pd.DataFrame(error_records).sort_values(["pid", "trial"]).reset_index(drop=True)
    error_df["prev_modify_magnitude"] = error_df.groupby("pid")["modify_magnitude"].shift(1)
    error_df["prev_action"] = error_df.groupby("pid")["action_label"].shift(1)
    error_df["prev_orig_eq_ai"] = error_df.groupby("pid")["orig_eq_ai"].shift(1)
    return error_df

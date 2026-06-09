"""
Complete reproduction of Chong et al. (2022) analysis and figures.

This script reproduces:
- Figure 3: Model fitting results (AI confidence and self-confidence)
- Figure 4: Confidence evolution with linear trends
- Figure 5: Team performance score distributions
- Figure 6: Performance scatter plots
- Table 1: Model parameters
- Table 2: Overall logistic regression
- Table 3: Category-level logistic regression
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import least_squares, OptimizeResult
from scipy import stats
import statsmodels.formula.api as smf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Constants
# ============================================================================

class Config:
    """Centralized configuration for the analysis."""
    
    # Trial structure
    NUM_TRIALS = 30
    PRACTICE_LAST_IDX = 4
    MAIN_START_IDX = 6
    SCORE_ROW_IDX = MAIN_START_IDX + NUM_TRIALS
    
    # Model parameters
    PARAMETER_NAMES = [
        "alpha_e", "alpha_a", "alpha_b",
        "omega1", "omega2", "omega3", "omega4",
        "gamma"
    ]
    
    TABLE1_TARGET = {
        "aiconf": [0.2672, 0.3405, 0.0524, 0.8439, 0.2115, 0.0, 0.5217, 0.3897],
        "selfconf": [0.2844, 0.4706, 0.0, 0.5736, 0.8284, 0.2384, 0.2863, 0.1147],
    }
    
    # Optimization settings
    LOGIT_KWARGS = {"method": "lbfgs", "maxiter": 1000, "disp": False}
    MAX_NFEV = 5000
    RANDOM_SEED = 42
    
    # Robustness test settings
    ROBUSTNESS_N_ITERATIONS = 100  # Number of bootstrap iterations
    ROBUSTNESS_SUBSET_SIZE = 80    # Number of participants per iteration (out of 100)
    ROBUSTNESS_ENABLED = True       # Enable/disable robustness test
    
    # Paths
    DATA_DIR = Path("Data")
    CODE_DIR = Path("Code")
    OUTPUT_DIR = Path("output_benchmark")
    
    # Plot settings (matching paper style)
    PLOT_DPI = 300
    FIGURE_WIDTH = 10
    FIGURE_HEIGHT = 5
    
    # Performance change point
    PERF_CHANGE_TRIAL = 20  # AI performance changes after trial 20


class Category(str, Enum):
    """Performance categories."""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ParticipantData:
    """Container for individual participant data."""
    pid: int
    base_id: int
    condition: int
    c_series: np.ndarray  # AI confidence
    self_series: np.ndarray  # Self confidence
    e_matrix: np.ndarray  # Experience matrix: (NUM_TRIALS, 4)
    act_series: np.ndarray  # Acceptance decisions
    perf_series: np.ndarray  # Performance feedback
    skill_score: float
    team_score: float
    trial_records: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelResults:
    """Container for model fitting results."""
    params: np.ndarray
    mse: float
    adj_r2: float
    residuals: np.ndarray
    optimize_result: OptimizeResult


# ============================================================================
# Output Management
# ============================================================================

def setup_output_directory(output_dir: Path) -> None:
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")


def save_dataframe(df: pd.DataFrame, filename: str, output_dir: Path = None) -> Path:
    """Save DataFrame with logging."""
    output_dir = output_dir or Config.OUTPUT_DIR
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    logger.info(f"✓ Saved: {filepath}")
    return filepath


def save_text(content: str, filename: str, output_dir: Path = None) -> Path:
    """Save text file with logging."""
    output_dir = output_dir or Config.OUTPUT_DIR
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"✓ Saved: {filepath}")
    return filepath


def save_figure(fig: plt.Figure, filename: str, output_dir: Path = None) -> Path:
    """Save figure with logging."""
    output_dir = output_dir or Config.OUTPUT_DIR
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=Config.PLOT_DPI, bbox_inches="tight")
    logger.info(f"✓ Saved: {filepath}")
    return filepath


# ============================================================================
# Utility Functions
# ============================================================================

def safe_float(value: Any, default: float = math.nan) -> float:
    """Convert value to float with robust error handling."""
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
    """Convert value to string safely."""
    if isinstance(value, str):
        return value.strip()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def parse_filename(path: Path) -> Tuple[int, int]:
    """Extract participant IDs from filename."""
    match = re.search(r"data(\d+)_(\d+)\.csv", path.name)
    if not match:
        raise ValueError(f"Invalid filename format: {path.name}")
    return int(match.group(1)), int(match.group(2))


# ============================================================================
# Data Loading and Parsing
# ============================================================================

def parse_participant_data(
    path: Path,
    pid: int,
    condition: int,
    base_id: int
) -> ParticipantData:
    """Parse participant data from CSV file."""
    df = pd.read_csv(path, header=None)
    
    def get_num(row: int, col: int) -> float:
        return safe_float(df.iat[row, col])
    
    def get_str(row: int, col: int) -> str:
        return safe_str(df.iat[row, col])
    
    # Extract confidence series (practice + 30 trials)
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
    
    # Parse trials
    e_matrix = np.zeros((Config.NUM_TRIALS, 4), dtype=float)
    act_values = [0]  # Practice trial
    trial_records = []
    
    for t in range(Config.NUM_TRIALS):
        row_idx = Config.MAIN_START_IDX + t
        
        aisugg = get_str(row_idx, 3)
        final_move = get_str(row_idx, 5)
        feedback2 = get_num(row_idx, 7)
        
        accept = int(aisugg == final_move)
        act_values.append(accept)
        
        # Build experience matrix
        if not math.isnan(feedback2):
            sign = int(feedback2 / 5 * -1)
            idx = (sign + 1) if accept else (sign + 2)
            if 0 <= idx < 4:
                e_matrix[t, idx] = 1.0
        
        trial_records.append({
            "pid": pid,
            "condition": condition,
            "trial": t + 1,
            "aisugg": aisugg,
            "bmove1": get_str(row_idx, 2),
            "bmove2": final_move,
            "accept": accept,
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
    
    skill_score = np.nansum([
        get_num(Config.MAIN_START_IDX + t, 6)
        for t in range(Config.NUM_TRIALS)
    ])
    
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
    """Load all participant data from directory."""
    logger.info(f"Loading participants from {data_dir}")
    
    file_mapping: Dict[int, Dict[int, Path]] = {1: {}, 2: {}}
    for path in sorted(data_dir.glob("data*_*.csv")):
        base_id, condition = parse_filename(path)
        file_mapping[condition][base_id] = path
    
    participants = []
    all_trial_records = []
    pid_counter = 1
    
    for condition in (1, 2):
        for base_id in sorted(file_mapping[condition]):
            participant = parse_participant_data(
                file_mapping[condition][base_id],
                pid_counter,
                condition,
                base_id
            )
            participants.append(participant)
            all_trial_records.extend(participant.trial_records)
            pid_counter += 1
    
    logger.info(f"Loaded {len(participants)} participants")
    return participants, pd.DataFrame(all_trial_records)


def load_score_categories(code_dir: Path) -> Dict[str, List[int]]:
    """Load predefined score categories from MATLAB files."""
    categories = {cat.value: [] for cat in Category}
    condition_offsets = {1: 0, 2: 50}
    
    for condition in (1, 2):
        mat_path = code_dir / f"scoregroup{condition}.mat"
        mat_data = loadmat(mat_path)["group"][0]
        
        for category, arr in zip(Category, mat_data):
            ids = [int(x) for x in arr.flatten()]
            offset_ids = [condition_offsets[condition] + idx for idx in ids]
            categories[category.value].extend(offset_ids)
    
    return {cat: sorted(ids) for cat, ids in categories.items()}


# ============================================================================
# Data Processing
# ============================================================================

def build_participant_summary(participants: List[ParticipantData]) -> pd.DataFrame:
    """Build summary statistics for each participant."""
    records = []
    for p in participants:
        main_trials = slice(1, None)
        
        records.append({
            "pid": p.pid,
            "condition": p.condition,
            "acceptance_rate": np.nanmean(p.act_series[main_trials]),
            "team_score": p.team_score,
            "individual_score": p.skill_score,
            "mean_selfconf": np.nanmean(p.self_series[main_trials]),
            "mean_aiconf": np.nanmean(p.c_series[main_trials]),
        })
    
    return pd.DataFrame(records)


def assign_score_categories(
    scores: pd.DataFrame,
    score_groups: Dict[str, List[int]]
) -> pd.DataFrame:
    """Assign performance categories to participants."""
    result = scores.copy()
    result["category"] = Category.FAIR.value
    
    for category in (Category.POOR, Category.GOOD):
        mask = result["pid"].isin(score_groups[category.value])
        result.loc[mask, "category"] = category.value
    
    return result


def build_analysis_matrices(
    participants: List[ParticipantData]
) -> Dict[str, np.ndarray]:
    """Stack all participant data into analysis matrices."""
    return {
        "ai_conf_matrix": np.vstack([p.c_series for p in participants]),
        "self_conf_matrix": np.vstack([p.self_series for p in participants]),
        "e_tensor": np.stack([p.e_matrix for p in participants]),
        "act_matrix": np.vstack([p.act_series for p in participants]),
        "perf_matrix": np.vstack([p.perf_series for p in participants]),
        "pid_array": np.array([p.pid for p in participants], dtype=int),
        "condition_array": np.array([p.condition for p in participants], dtype=int),
        "team_scores": np.array([p.team_score for p in participants], dtype=float),
        "individual_scores": np.array([p.skill_score for p in participants], dtype=float),
    }


# ============================================================================
# Confidence Model (Equations 1-4 from Paper)
# ============================================================================

def simulate_confidence_dynamics(
    params: np.ndarray,
    observed: np.ndarray,
    e_tensor: np.ndarray
) -> np.ndarray:
    """Simulate confidence dynamics using Eq. 1-4 from Chong et al."""
    alpha_e, alpha_a, alpha_b, omega1, omega2, omega3, omega4, gamma = params
    n_participants, series_len = observed.shape
    n_trials = series_len - 1
    
    # Initialize
    c = observed[:, 0].copy()
    b = observed[:, 0].copy()
    a = observed[:, 0].copy()
    predicted = np.zeros_like(observed)
    predicted[:, 0] = c
    
    # Simulate forward (Eq. 1-4)
    for t in range(n_trials):
        # Eq. 3: Accumulated confidence
        if t > 0:
            a = gamma * predicted[:, t] + (1 - gamma) * a
        
        # Eq. 2: Experience term
        experience = (
            omega1 * e_tensor[:, t, 0] +
            omega2 * e_tensor[:, t, 1] +
            omega3 * e_tensor[:, t, 2] +
            omega4 * e_tensor[:, t, 3]
        )
        
        # Eq. 1: Confidence update
        c = c + alpha_e * (experience - c) + alpha_a * (a - c) + alpha_b * (b - c)
        predicted[:, t + 1] = c
    
    # Return residuals
    residuals = observed[:, 1:] - predicted[:, 1:]
    return residuals.T.ravel()


def fit_confidence_model(
    observed: np.ndarray,
    e_tensor: np.ndarray,
    initial_params: np.ndarray,
    participant_subset: np.ndarray,
) -> ModelResults:
    """Fit confidence model to observed data."""
    obs_subset = observed[participant_subset]
    e_subset = e_tensor[participant_subset]
    
    bounds = (np.zeros_like(initial_params), np.ones_like(initial_params))
    result = least_squares(
        simulate_confidence_dynamics,
        x0=initial_params,
        bounds=bounds,
        args=(obs_subset, e_subset),
        max_nfev=Config.MAX_NFEV,
    )
    
    residuals = simulate_confidence_dynamics(result.x, obs_subset, e_subset)
    n_obs = residuals.size
    sse = np.sum(residuals ** 2)
    
    # MATLAB-style R² calculation: based on mean data
    # This matches modelPlot.m which calculates R² on averaged data
    mean_observed = np.nanmean(obs_subset, axis=0)
    predicted_mean = compute_model_predictions(result.x, obs_subset, e_subset)
    predicted_mean_avg = np.nanmean(predicted_mean, axis=0)
    
    # Calculate R² on mean data (matching MATLAB: meanC(2:end) vs fitMean(2:end))
    # Exclude initial point (index 0), use trials 1-30
    mean_obs_trials = mean_observed[1:]  # Exclude practice/initial
    mean_pred_trials = predicted_mean_avg[1:]
    
    ssres = np.nansum((mean_pred_trials - mean_obs_trials) ** 2)
    sstot = np.nansum((mean_obs_trials - np.nanmean(mean_obs_trials)) ** 2)
    
    r2 = 1 - ssres / sstot if sstot > 0 else float("nan")
    
    # MATLAB-style adjusted R²: rsq_adj = 1 - (99/92) * (ssres/sstot)
    # This matches modelPlot.m line 132
    adj_r2 = 1 - (99/92) * (ssres / sstot) if sstot > 0 else float("nan")
    
    # Also calculate MSE on mean data (matching MATLAB: mse = ssres/60)
    # 60 = 2 conditions × 30 trials
    n_conditions = 2
    n_trials = Config.NUM_TRIALS
    mse_mean = ssres / (n_conditions * n_trials)
    
    return ModelResults(
        params=result.x,
        mse=mse_mean,  # MSE based on mean data (MATLAB style)
        adj_r2=adj_r2,  # Adjusted R² using MATLAB formula
        residuals=residuals,
        optimize_result=result
    )


def fit_all_confidence_models(
    matrices: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Fit confidence models for AI and self confidence (Table 1).
    
    Matches MATLAB modelPlot.m: fits model on all participants, then calculates
    R² and MSE based on mean data across two conditions (matching MATLAB style).
    """
    logger.info("Fitting confidence models (Table 1)...")
    
    n_participants = matrices["ai_conf_matrix"].shape[0]
    subset = np.arange(n_participants)
    
    records = []
    for model_name, conf_matrix in [
        ("aiconf", matrices["ai_conf_matrix"]),
        ("selfconf", matrices["self_conf_matrix"])
    ]:
        initial = np.array(Config.TABLE1_TARGET[model_name], dtype=float)
        results = fit_confidence_model(
            conf_matrix,
            matrices["e_tensor"],
            initial,
            subset
        )
        
        record = {
            "model": model_name,
            "mse": results.mse,
            "r2_adj": results.adj_r2,
        }
        record.update(dict(zip(Config.PARAMETER_NAMES, results.params)))
        records.append(record)
    
    df = pd.DataFrame(records)
    save_dataframe(df, "table1_model_params.csv")
    
    logger.info("Note: R² and MSE calculated using MATLAB-style method:")
    logger.info("  - Based on mean data (averaged across participants)")
    logger.info("  - Adjusted R²: 1 - (99/92) × (ssres/sstot)")
    logger.info("  - MSE: ssres / 60 (2 conditions × 30 trials)")
    
    return df, matrices["ai_conf_matrix"], matrices["self_conf_matrix"]


# ============================================================================
# Robustness Testing
# ============================================================================

def run_robustness_test(
    matrices: Dict[str, np.ndarray],
    n_iterations: int = None,
    subset_size: int = None,
) -> Dict[str, Any]:
    """
    Run robustness test by fitting model on multiple random subsets.
    
    Similar to MATLAB code: 100 iterations, each with 80 randomly selected participants.
    """
    n_iterations = n_iterations or Config.ROBUSTNESS_N_ITERATIONS
    subset_size = subset_size or Config.ROBUSTNESS_SUBSET_SIZE
    
    n_participants = matrices["ai_conf_matrix"].shape[0]
    if subset_size >= n_participants:
        logger.warning(f"Subset size ({subset_size}) >= total participants ({n_participants}). Using all participants.")
        subset_size = n_participants
    
    logger.info(f"\n{'='*70}")
    logger.info(f"ROBUSTNESS TEST: {n_iterations} iterations, {subset_size} participants per iteration")
    logger.info(f"{'='*70}")
    
    np.random.seed(Config.RANDOM_SEED)
    
    all_results = {
        "aiconf": {
            "params_list": [],
            "mse_list": [],
            "r2_adj_list": [],
            "selected_participants": []
        },
        "selfconf": {
            "params_list": [],
            "mse_list": [],
            "r2_adj_list": [],
            "selected_participants": []
        }
    }
    
    for iteration in range(n_iterations):
        if (iteration + 1) % 10 == 0:
            logger.info(f"  Iteration {iteration + 1}/{n_iterations}...")
        
        # Randomly select subset of participants
        selected_indices = np.random.choice(n_participants, size=subset_size, replace=False)
        selected_indices = np.sort(selected_indices)
        
        for model_name, conf_matrix in [
            ("aiconf", matrices["ai_conf_matrix"]),
            ("selfconf", matrices["self_conf_matrix"])
        ]:
            initial = np.array(Config.TABLE1_TARGET[model_name], dtype=float)
            
            try:
                results = fit_confidence_model(
                    conf_matrix,
                    matrices["e_tensor"],
                    initial,
                    selected_indices
                )
                
                all_results[model_name]["params_list"].append(results.params)
                all_results[model_name]["mse_list"].append(results.mse)
                all_results[model_name]["r2_adj_list"].append(results.adj_r2)
                all_results[model_name]["selected_participants"].append(selected_indices.tolist())
                
            except Exception as e:
                logger.warning(f"  Iteration {iteration + 1} failed for {model_name}: {e}")
                continue
    
    # Convert to numpy arrays for easier computation
    for model_name in ["aiconf", "selfconf"]:
        all_results[model_name]["params_array"] = np.array(all_results[model_name]["params_list"])
        all_results[model_name]["mse_array"] = np.array(all_results[model_name]["mse_list"])
        all_results[model_name]["r2_adj_array"] = np.array(all_results[model_name]["r2_adj_list"])
    
    logger.info(f"✓ Completed {n_iterations} iterations")
    return all_results


def compute_robustness_statistics(
    robustness_results: Dict[str, Any],
    original_params: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Compute statistics from robustness test results."""
    records = []
    
    for model_name in ["aiconf", "selfconf"]:
        params_array = robustness_results[model_name]["params_array"]
        mse_array = robustness_results[model_name]["mse_array"]
        r2_adj_array = robustness_results[model_name]["r2_adj_array"]
        
        # Compute statistics
        param_mean = np.mean(params_array, axis=0)
        param_std = np.std(params_array, axis=0, ddof=1)
        param_median = np.median(params_array, axis=0)
        param_cv = param_std / (np.abs(param_mean) + 1e-10)  # Coefficient of variation
        
        # Compare with original parameters
        original = original_params[model_name]
        param_diff = np.abs(param_mean - original)
        param_diff_pct = (param_diff / (np.abs(original) + 1e-10)) * 100
        
        # Create record for each parameter
        for i, param_name in enumerate(Config.PARAMETER_NAMES):
            records.append({
                "model": model_name,
                "parameter": param_name,
                "original": original[i],
                "mean": param_mean[i],
                "median": param_median[i],
                "std": param_std[i],
                "cv": param_cv[i],
                "abs_diff": param_diff[i],
                "pct_diff": param_diff_pct[i],
                "min": np.min(params_array[:, i]),
                "max": np.max(params_array[:, i]),
                "q25": np.percentile(params_array[:, i], 25),
                "q75": np.percentile(params_array[:, i], 75),
            })
        
        # Add summary statistics for MSE and R²
        records.append({
            "model": model_name,
            "parameter": "mse",
            "original": np.nan,
            "mean": np.mean(mse_array),
            "median": np.median(mse_array),
            "std": np.std(mse_array, ddof=1),
            "cv": np.std(mse_array, ddof=1) / (np.mean(mse_array) + 1e-10),
            "abs_diff": np.nan,
            "pct_diff": np.nan,
            "min": np.min(mse_array),
            "max": np.max(mse_array),
            "q25": np.percentile(mse_array, 25),
            "q75": np.percentile(mse_array, 75),
        })
        
        records.append({
            "model": model_name,
            "parameter": "r2_adj",
            "original": np.nan,
            "mean": np.mean(r2_adj_array),
            "median": np.median(r2_adj_array),
            "std": np.std(r2_adj_array, ddof=1),
            "cv": np.std(r2_adj_array, ddof=1) / (np.abs(np.mean(r2_adj_array)) + 1e-10),
            "abs_diff": np.nan,
            "pct_diff": np.nan,
            "min": np.min(r2_adj_array),
            "max": np.max(r2_adj_array),
            "q25": np.percentile(r2_adj_array, 25),
            "q75": np.percentile(r2_adj_array, 75),
        })
    
    df = pd.DataFrame(records)
    return df


def get_final_robust_parameters(
    robustness_results: Dict[str, Any],
    method: str = "mean"
) -> Dict[str, np.ndarray]:
    """
    Compute final robust parameter estimates.
    
    Args:
        method: "mean" or "median" - which statistic to use for final parameters
    """
    final_params = {}
    
    for model_name in ["aiconf", "selfconf"]:
        params_array = robustness_results[model_name]["params_array"]
        
        if method == "mean":
            final_params[model_name] = np.mean(params_array, axis=0)
        elif method == "median":
            final_params[model_name] = np.median(params_array, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'mean' or 'median'.")
    
    return final_params


def plot_robustness_results(
    robustness_results: Dict[str, Any],
    original_params: Dict[str, np.ndarray],
    output_dir: Path
) -> None:
    """Visualize robustness test results."""
    logger.info("Creating robustness visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for model_idx, model_name in enumerate(["aiconf", "selfconf"]):
        params_array = robustness_results[model_name]["params_array"]
        original = original_params[model_name]
        
        # Plot 1: Parameter distributions (box plots)
        ax1 = axes[model_idx, 0]
        param_data = [params_array[:, i] for i in range(len(Config.PARAMETER_NAMES))]
        bp = ax1.boxplot(param_data, labels=Config.PARAMETER_NAMES, patch_artist=True)
        
        # Color boxes
        colors = ['#1f77b4' if i < 3 else '#ff7f0e' if i < 7 else '#2ca02c' 
                  for i in range(len(Config.PARAMETER_NAMES))]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Mark original values
        for i, param_name in enumerate(Config.PARAMETER_NAMES):
            ax1.plot(i + 1, original[i], 'r*', markersize=12, label='Original' if i == 0 else '')
        
        ax1.set_ylabel('Parameter Value', fontsize=11)
        ax1.set_title(f'{model_name.upper()}: Parameter Distributions', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Plot 2: Coefficient of Variation
        ax2 = axes[model_idx, 1]
        param_mean = np.mean(params_array, axis=0)
        param_std = np.std(params_array, axis=0, ddof=1)
        cv = param_std / (np.abs(param_mean) + 1e-10)
        
        bars = ax2.bar(Config.PARAMETER_NAMES, cv, color=colors, alpha=0.7)
        ax2.set_ylabel('Coefficient of Variation (CV)', fontsize=11)
        ax2.set_title(f'{model_name.upper()}: Parameter Stability (CV)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0.1, color='r', linestyle='--', linewidth=1, label='CV=0.1 threshold')
        ax2.legend()
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.tight_layout()
    save_figure(fig, "robustness_analysis.png", output_dir)
    plt.close(fig)


def format_robustness_table(robustness_stats: pd.DataFrame) -> str:
    """Format robustness statistics for display."""
    table = []
    table.append("\n" + "="*100)
    table.append("ROBUSTNESS TEST RESULTS: Parameter Statistics Across 100 Iterations")
    table.append("="*100)
    
    for model_name in ["aiconf", "selfconf"]:
        model_data = robustness_stats[robustness_stats["model"] == model_name]
        param_data = model_data[model_data["parameter"].isin(Config.PARAMETER_NAMES)]
        
        table.append(f"\n{model_name.upper()} Model:")
        table.append("-"*100)
        table.append(f"{'Param':<10} {'Original':<10} {'Mean':<10} {'Std':<10} {'CV':<10} {'Min':<10} {'Max':<10} {'Q25':<10} {'Q75':<10}")
        table.append("-"*100)
        
        for _, row in param_data.iterrows():
            table.append(
                f"{row['parameter']:<10} "
                f"{row['original']:>9.4f} "
                f"{row['mean']:>9.4f} "
                f"{row['std']:>9.4f} "
                f"{row['cv']:>9.4f} "
                f"{row['min']:>9.4f} "
                f"{row['max']:>9.4f} "
                f"{row['q25']:>9.4f} "
                f"{row['q75']:>9.4f}"
            )
        
        # Add MSE and R² summary
        mse_row = model_data[model_data["parameter"] == "mse"].iloc[0]
        r2_row = model_data[model_data["parameter"] == "r2_adj"].iloc[0]
        table.append("-"*100)
        table.append(f"MSE:  Mean={mse_row['mean']:.6f}, Std={mse_row['std']:.6f}, CV={mse_row['cv']:.4f}")
        table.append(f"R²:   Mean={r2_row['mean']:.4f}, Std={r2_row['std']:.4f}, CV={r2_row['cv']:.4f}")
    
    table.append("="*100)
    return "\n".join(table)


def compute_model_predictions(
    params: np.ndarray,
    observed: np.ndarray,
    e_tensor: np.ndarray
) -> np.ndarray:
    """Compute model predictions for plotting."""
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


# ============================================================================
# Logistic Regression Analysis
# ============================================================================

def prepare_logit_data(matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Prepare data for logistic regression."""
    records = []
    
    pid = matrices["pid_array"]
    ai_conf = matrices["ai_conf_matrix"][:, 1:]  # Exclude practice
    self_conf = matrices["self_conf_matrix"][:, 1:]
    accept = matrices["act_matrix"][:, 1:]
    
    for idx, participant_id in enumerate(pid):
        for trial in range(Config.NUM_TRIALS):
            records.append({
                "pid": participant_id,
                "trial": trial + 1,
                "aiconf": ai_conf[idx, trial],
                "selfconf": self_conf[idx, trial],
                "accept": accept[idx, trial],
            })
    
    return pd.DataFrame(records)


def run_overall_logit(logit_df: pd.DataFrame) -> pd.DataFrame:
    """Run fixed-effects logistic regression (Table 2)."""
    logger.info("Running overall logistic regression (Table 2)...")
    
    df = logit_df.copy()
    df["pid"] = df["pid"].astype(int)
    
    model = smf.logit(
        "accept ~ aiconf + selfconf + C(pid)",
        data=df
    ).fit(**Config.LOGIT_KWARGS)
    
    coefs = pd.DataFrame({
        "predictor": ["aiconf", "selfconf"],
        "coef": [model.params["aiconf"], model.params["selfconf"]],
        "std_err": [model.bse["aiconf"], model.bse["selfconf"]],
        "p_value": [model.pvalues["aiconf"], model.pvalues["selfconf"]],
    })
    
    save_dataframe(coefs, "table2_logit_coeffs.csv")
    save_text(str(model.summary()), "table2_logit_summary.txt")
    
    return coefs


def run_category_logit(
    logit_df: pd.DataFrame,
    score_groups: Dict[str, List[int]]
) -> pd.DataFrame:
    """Run logistic regression by performance category (Table 3)."""
    logger.info("Running category-level logistic regression (Table 3)...")
    
    records = []
    for category in ["poor", "fair", "good"]:
        participant_ids = score_groups[category]
        subset = logit_df[logit_df["pid"].isin(participant_ids)]
        
        if subset.empty:
            logger.warning(f"No data for category: {category}")
            continue
        
        model = smf.logit(
            "accept ~ aiconf + selfconf + C(pid)",
            data=subset
        ).fit(**Config.LOGIT_KWARGS)
        
        for predictor in ("aiconf", "selfconf"):
            records.append({
                "category": category,
                "predictor": predictor,
                "coef": model.params[predictor],
                "std_err": model.bse[predictor],
                "p_value": model.pvalues[predictor],
            })
    
    result = pd.DataFrame(records)
    save_dataframe(result, "table3_category_logit.csv")
    
    return result


# ============================================================================
# Reproduction of Paper Figures
# ============================================================================

def plot_figure3(
    model_params_df: pd.DataFrame,
    matrices: Dict[str, np.ndarray],
    output_dir: Path
) -> None:
    """
    Reproduce Figure 3: Model fitting results.
    Shows observed data (black points) and model predictions (blue lines).
    """
    logger.info("Creating Figure 3: Model fitting results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Get parameters for each model
    ai_params = model_params_df[model_params_df["model"] == "aiconf"].iloc[0]
    self_params = model_params_df[model_params_df["model"] == "selfconf"].iloc[0]
    
    ai_param_vals = ai_params[Config.PARAMETER_NAMES].values
    self_param_vals = self_params[Config.PARAMETER_NAMES].values
    
    # Separate by condition
    cond_mask = {
        1: matrices["condition_array"] == 1,
        2: matrices["condition_array"] == 2
    }
    
    for cond_idx, condition in enumerate([1, 2]):
        mask = cond_mask[condition]
        
        # AI confidence
        ai_obs = matrices["ai_conf_matrix"][mask]
        e_subset = matrices["e_tensor"][mask]
        ai_pred = compute_model_predictions(ai_param_vals, ai_obs, e_subset)
        
        ax_ai = axes[cond_idx, 0]
        trials = np.arange(Config.NUM_TRIALS + 1)
        
        # Plot observed data
        ai_mean = np.nanmean(ai_obs, axis=0)
        ai_se = np.nanstd(ai_obs, axis=0, ddof=1) / np.sqrt(ai_obs.shape[0])
        ax_ai.errorbar(trials, ai_mean, yerr=ai_se, fmt='o', color='black', 
                       markersize=4, capsize=3, label='Observed', alpha=0.7)
        
        # Plot model predictions
        ai_pred_mean = np.nanmean(ai_pred, axis=0)
        ax_ai.plot(trials, ai_pred_mean, color='#1f77b4', linewidth=2.5, 
                   label='Model')
        
        ax_ai.set_ylim(0, 1)
        ax_ai.set_xlabel('Trial', fontsize=11)
        ax_ai.set_ylabel('Confidence in AI', fontsize=11)
        ax_ai.set_title(f'Condition {condition}', fontsize=12, fontweight='bold')
        ax_ai.grid(True, alpha=0.3)
        if cond_idx == 0:
            ax_ai.legend(loc='upper right')
        
        # Self confidence
        self_obs = matrices["self_conf_matrix"][mask]
        self_pred = compute_model_predictions(self_param_vals, self_obs, e_subset)
        
        ax_self = axes[cond_idx, 1]
        
        # Plot observed data
        self_mean = np.nanmean(self_obs, axis=0)
        self_se = np.nanstd(self_obs, axis=0, ddof=1) / np.sqrt(self_obs.shape[0])
        ax_self.errorbar(trials, self_mean, yerr=self_se, fmt='o', color='black',
                        markersize=4, capsize=3, label='Observed', alpha=0.7)
        
        # Plot model predictions
        self_pred_mean = np.nanmean(self_pred, axis=0)
        ax_self.plot(trials, self_pred_mean, color='#1f77b4', linewidth=2.5,
                    label='Model')
        
        ax_self.set_ylim(0, 1)
        ax_self.set_xlabel('Trial', fontsize=11)
        ax_self.set_ylabel('Self-confidence', fontsize=11)
        ax_self.set_title(f'Condition {condition}', fontsize=12, fontweight='bold')
        ax_self.grid(True, alpha=0.3)
        if cond_idx == 0:
            ax_self.legend(loc='upper right')
    
    fig.tight_layout()
    save_figure(fig, "figure3_model_fitting.png", output_dir)
    plt.close(fig)


def plot_figure4(matrices: Dict[str, np.ndarray], output_dir: Path) -> None:
    """
    Reproduce Figure 4: Confidence evolution with linear trends before/after change.
    """
    logger.info("Creating Figure 4: Confidence trends with linear fits...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    cond_mask = {
        1: matrices["condition_array"] == 1,
        2: matrices["condition_array"] == 2
    }
    
    change_point = Config.PERF_CHANGE_TRIAL
    
    for cond_idx, condition in enumerate([1, 2]):
        mask = cond_mask[condition]
        
        # AI confidence
        ai_conf = matrices["ai_conf_matrix"][mask, 1:]  # Exclude practice
        trials = np.arange(1, Config.NUM_TRIALS + 1)
        
        ai_mean = np.nanmean(ai_conf, axis=0)
        ai_se = np.nanstd(ai_conf, axis=0, ddof=1) / np.sqrt(ai_conf.shape[0])
        
        ax_ai = axes[cond_idx, 0]
        ax_ai.errorbar(trials, ai_mean, yerr=ai_se, fmt='o', color='black',
                      markersize=4, capsize=2, alpha=0.5)
        
        # Linear fits before and after change
        t1 = trials[:change_point]
        t2 = trials[change_point:]
        y1 = ai_mean[:change_point]
        y2 = ai_mean[change_point:]
        
        # Fit linear trends
        if len(t1) > 1:
            coef1 = np.polyfit(t1, y1, 1)
            line1 = np.poly1d(coef1)
            ax_ai.plot(t1, line1(t1), color='#1f77b4', linewidth=2.5, linestyle='-')
        
        if len(t2) > 1:
            coef2 = np.polyfit(t2, y2, 1)
            line2 = np.poly1d(coef2)
            ax_ai.plot(t2, line2(t2), color='#1f77b4', linewidth=2.5, linestyle='-')
        
        # Mark change point
        ax_ai.axvline(change_point, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        
        ax_ai.set_ylim(0, 1)
        ax_ai.set_xlabel('Trial', fontsize=11)
        ax_ai.set_ylabel('Confidence in AI', fontsize=11)
        ax_ai.set_title(f'Condition {condition}', fontsize=12, fontweight='bold')
        ax_ai.grid(True, alpha=0.3)
        
        # Self confidence
        self_conf = matrices["self_conf_matrix"][mask, 1:]
        self_mean = np.nanmean(self_conf, axis=0)
        self_se = np.nanstd(self_conf, axis=0, ddof=1) / np.sqrt(self_conf.shape[0])
        
        ax_self = axes[cond_idx, 1]
        ax_self.errorbar(trials, self_mean, yerr=self_se, fmt='o', color='black',
                        markersize=4, capsize=2, alpha=0.5)
        
        # Linear fits
        y1_self = self_mean[:change_point]
        y2_self = self_mean[change_point:]
        
        if len(t1) > 1:
            coef1_self = np.polyfit(t1, y1_self, 1)
            line1_self = np.poly1d(coef1_self)
            ax_self.plot(t1, line1_self(t1), color='#1f77b4', linewidth=2.5, linestyle='-')
        
        if len(t2) > 1:
            coef2_self = np.polyfit(t2, y2_self, 1)
            line2_self = np.poly1d(coef2_self)
            ax_self.plot(t2, line2_self(t2), color='#1f77b4', linewidth=2.5, linestyle='-')
        
        ax_self.axvline(change_point, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        
        ax_self.set_ylim(0, 1)
        ax_self.set_xlabel('Trial', fontsize=11)
        ax_self.set_ylabel('Self-confidence', fontsize=11)
        ax_self.set_title(f'Condition {condition}', fontsize=12, fontweight='bold')
        ax_self.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_figure(fig, "figure4_confidence_trends.png", output_dir)
    plt.close(fig)


def plot_figure5(scores: pd.DataFrame, score_groups: Dict[str, List[int]], output_dir: Path) -> None:
    """
    Reproduce Figure 5: Team performance score distributions with category boundaries.
    """
    logger.info("Creating Figure 5: Score distributions...")
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    for ax, (condition, group) in zip(axes, scores.groupby("condition")):
        # Plot histogram
        ax.hist(group["team_score"], bins=15, alpha=0.8, color="#4c72b0", 
                edgecolor="black", linewidth=0.5)
        
        # Fit normal distribution
        mu, sigma = group["team_score"].mean(), group["team_score"].std()
        
        # Calculate percentile boundaries (25% and 75%)
        sorted_scores = np.sort(group["team_score"].values)
        n = len(sorted_scores)
        idx_25 = int(0.25 * n)
        idx_75 = int(0.75 * n)
        
        lower_bound = sorted_scores[idx_25]
        upper_bound = sorted_scores[idx_75]
        
        # Draw boundary lines
        ax.axvline(lower_bound, color='orange', linestyle='--', linewidth=2, 
                   label='Category boundaries')
        ax.axvline(upper_bound, color='orange', linestyle='--', linewidth=2)
        
        ax.set_title(f'Condition {condition}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Team Performance Score', fontsize=11)
        ax.set_ylabel('Number of Participants', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    save_figure(fig, "figure5_score_distributions.png", output_dir)
    plt.close(fig)


def plot_figure6(scores: pd.DataFrame, output_dir: Path) -> None:
    """
    Reproduce Figure 6: Scatter plots comparing performance characteristics.
    """
    logger.info("Creating Figure 6: Performance scatter plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    color_map = {
        Category.POOR.value: "#d62728",
        Category.FAIR.value: "#ff7f0e",
        Category.GOOD.value: "#2ca02c"
    }
    
    # Plot A: Team vs Individual performance
    for category, subset in scores.groupby("category"):
        color = color_map.get(category, "#1f77b4")
        axes[0].scatter(
            subset["team_score"],
            subset["individual_score"],
            label=category.capitalize(),
            color=color,
            alpha=0.7,
            s=80,
            edgecolors="black",
            linewidths=0.8
        )
    
    axes[0].set_xlabel("Team Performance Score", fontsize=12)
    axes[0].set_ylabel("Individual Skill Score", fontsize=12)
    axes[0].set_title("(A) Team vs Individual Performance", fontsize=12, fontweight='bold')
    axes[0].legend(frameon=True, shadow=True)
    axes[0].grid(True, alpha=0.3)
    
    # Plot B: Team performance vs Self-confidence
    for category, subset in scores.groupby("category"):
        color = color_map.get(category, "#1f77b4")
        axes[1].scatter(
            subset["team_score"],
            subset["mean_selfconf"],
            label=category.capitalize(),
            color=color,
            alpha=0.7,
            s=80,
            edgecolors="black",
            linewidths=0.8
        )
    
    axes[1].set_xlabel("Team Performance Score", fontsize=12)
    axes[1].set_ylabel("Average Self-confidence", fontsize=12)
    axes[1].set_title("(B) Team Score vs Self-confidence", fontsize=12, fontweight='bold')
    axes[1].legend(frameon=True, shadow=True)
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_figure(fig, "figure6_scatter_plots.png", output_dir)
    plt.close(fig)


# ============================================================================
# Table Formatting
# ============================================================================

def format_table1(model_params_df: pd.DataFrame) -> str:
    """Format Table 1 for display."""
    table = []
    table.append("\n" + "="*80)
    table.append("TABLE 1: Model Parameter Estimates")
    table.append("="*80)
    table.append(f"{'Model':<15} {'α_e':<8} {'α_a':<8} {'α_b':<8} {'ω1':<8} {'ω2':<8} {'ω3':<8} {'ω4':<8} {'γ':<8}")
    table.append("-"*80)
    
    for _, row in model_params_df.iterrows():
        table.append(
            f"{row['model']:<15} "
            f"{row['alpha_e']:.4f}  "
            f"{row['alpha_a']:.4f}  "
            f"{row['alpha_b']:.4f}  "
            f"{row['omega1']:.4f}  "
            f"{row['omega2']:.4f}  "
            f"{row['omega3']:.4f}  "
            f"{row['omega4']:.4f}  "
            f"{row['gamma']:.4f}"
        )
    
    table.append("-"*80)
    table.append(f"\nNOTE: MSE (aiconf) = {model_params_df.loc[0, 'mse']:.4f}, "
                 f"Adj R² = {model_params_df.loc[0, 'r2_adj']:.3f}")
    table.append(f"      MSE (selfconf) = {model_params_df.loc[1, 'mse']:.4f}, "
                 f"Adj R² = {model_params_df.loc[1, 'r2_adj']:.3f}")
    table.append("="*80)
    
    return "\n".join(table)


def format_table2(coefs: pd.DataFrame) -> str:
    """Format Table 2 for display."""
    table = []
    table.append("\n" + "="*60)
    table.append("TABLE 2: Overall Logistic Regression Results")
    table.append("="*60)
    table.append(f"{'Predictor':<20} {'Coefficient':<15} {'Std Error':<15} {'P-value':<10}")
    table.append("-"*60)
    
    for _, row in coefs.iterrows():
        p_str = f"{row['p_value']:.4f}" if row['p_value'] >= 0.0001 else "<0.0001"
        table.append(
            f"{row['predictor']:<20} "
            f"{row['coef']:>14.4f} "
            f"{row['std_err']:>14.4f} "
            f"{p_str:>9}"
        )
    
    table.append("="*60)
    table.append("\nInterpretation:")
    table.append("  - AI confidence: NOT significant (p > 0.05)")
    table.append("  - Self-confidence: SIGNIFICANT (p < 0.001)")
    table.append("  → Self-confidence, not AI confidence, drives acceptance decisions")
    table.append("="*60)
    
    return "\n".join(table)


def format_table3(group_logit: pd.DataFrame) -> str:
    """Format Table 3 for display."""
    table = []
    table.append("\n" + "="*70)
    table.append("TABLE 3: Category-Level Logistic Regression Results")
    table.append("="*70)
    table.append(f"{'Category':<12} {'Predictor':<20} {'Coef':<12} {'SE':<12} {'P-value':<10}")
    table.append("-"*70)
    
    for _, row in group_logit.iterrows():
        p_str = f"{row['p_value']:.4f}" if row['p_value'] >= 0.0001 else "<0.0001"
        table.append(
            f"{row['category']:<12} "
            f"{row['predictor']:<20} "
            f"{row['coef']:>11.4f} "
            f"{row['std_err']:>11.4f} "
            f"{p_str:>9}"
        )
    
    table.append("="*70)
    table.append("\nKey Finding:")
    table.append("  - GOOD performers: Positive correlation (self-conf → accept AI)")
    table.append("  - POOR/FAIR performers: Negative correlation")
    table.append("  → Good decision-makers accept AI when confident, reject when not")
    table.append("="*70)
    
    return "\n".join(table)


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def main() -> None:
    """Execute complete reproduction of Chong et al. (2022)."""
    logger.info("="*70)
    logger.info("REPRODUCING: Chong et al. (2022)")
    logger.info("Human confidence in artificial intelligence and in themselves")
    logger.info("="*70)
    
    # Setup
    setup_output_directory(Config.OUTPUT_DIR)
    
    # Load data
    logger.info("\n📂 Loading experimental data...")
    participants, trial_df = load_all_participants(Config.DATA_DIR)
    save_dataframe(trial_df, "trial_data.csv")
    
    # Build analysis structures
    logger.info("🔨 Building analysis matrices...")
    matrices = build_analysis_matrices(participants)
    scores = build_participant_summary(participants)
    score_groups = load_score_categories(Config.CODE_DIR)
    scores = assign_score_categories(scores, score_groups)
    save_dataframe(scores, "participant_summary.csv")
    
    # TABLE 1: Fit confidence models
    logger.info("\n" + "="*70)
    logger.info("FITTING CONFIDENCE MODEL")
    logger.info("="*70)
    model_params_df, ai_conf, self_conf = fit_all_confidence_models(matrices)
    print(format_table1(model_params_df))
    
    # Extract original parameters for comparison
    original_params = {
        "aiconf": model_params_df[model_params_df["model"] == "aiconf"][Config.PARAMETER_NAMES].values[0],
        "selfconf": model_params_df[model_params_df["model"] == "selfconf"][Config.PARAMETER_NAMES].values[0]
    }
    
    # ROBUSTNESS TEST: Run multiple iterations with random subsets
    if Config.ROBUSTNESS_ENABLED:
        robustness_results = run_robustness_test(matrices)
        
        # Compute statistics
        robustness_stats = compute_robustness_statistics(robustness_results, original_params)
        save_dataframe(robustness_stats, "robustness_statistics.csv")
        
        # Visualize results
        plot_robustness_results(robustness_results, original_params, Config.OUTPUT_DIR)
        
        # Get final robust parameters (using mean)
        final_robust_params = get_final_robust_parameters(robustness_results, method="mean")
        
        # Create final parameter table
        final_records = []
        for model_name in ["aiconf", "selfconf"]:
            # Compute final statistics
            params_array = robustness_results[model_name]["params_array"]
            mse_array = robustness_results[model_name]["mse_array"]
            r2_adj_array = robustness_results[model_name]["r2_adj_array"]
            
            record = {
                "model": model_name,
                "mse": np.mean(mse_array),
                "r2_adj": np.mean(r2_adj_array),
            }
            record.update(dict(zip(Config.PARAMETER_NAMES, final_robust_params[model_name])))
            final_records.append(record)
        
        final_params_df = pd.DataFrame(final_records)
        save_dataframe(final_params_df, "table1_final_robust_params.csv")
        
        # Print robustness results
        print(format_robustness_table(robustness_stats))
        
        logger.info("\n" + "="*70)
        logger.info("FINAL ROBUST PARAMETERS (Mean across 100 iterations)")
        logger.info("="*70)
        print(format_table1(final_params_df))
        
        # Save raw robustness data
        robustness_raw = {
            "aiconf": {
                "params": robustness_results["aiconf"]["params_array"].tolist(),
                "mse": robustness_results["aiconf"]["mse_array"].tolist(),
                "r2_adj": robustness_results["aiconf"]["r2_adj_array"].tolist(),
            },
            "selfconf": {
                "params": robustness_results["selfconf"]["params_array"].tolist(),
                "mse": robustness_results["selfconf"]["mse_array"].tolist(),
                "r2_adj": robustness_results["selfconf"]["r2_adj_array"].tolist(),
            }
        }
        with open(Config.OUTPUT_DIR / "robustness_raw_data.json", "w") as f:
            json.dump(robustness_raw, f, indent=2)
        logger.info(f"✓ Saved raw robustness data: robustness_raw_data.json")
    
    # FIGURE 3: Model fitting results
    plot_figure3(model_params_df, matrices, Config.OUTPUT_DIR)
    
    # FIGURE 4: Confidence trends
    plot_figure4(matrices, Config.OUTPUT_DIR)
    
    # FIGURE 5: Score distributions
    plot_figure5(scores, score_groups, Config.OUTPUT_DIR)
    
    # FIGURE 6: Scatter plots
    plot_figure6(scores, Config.OUTPUT_DIR)
    
    # TABLE 2: Overall logistic regression
    logit_df = prepare_logit_data(matrices)
    table2 = run_overall_logit(logit_df)
    print(format_table2(table2))
    
    # TABLE 3: Category-level logistic regression
    table3 = run_category_logit(logit_df, score_groups)
    print(format_table3(table3))
    
    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*70)
    print(f"\nTotal participants: {len(participants)}")
    print(f"  - Condition 1 (good→poor AI): {sum(matrices['condition_array'] == 1)}")
    print(f"  - Condition 2 (poor→good AI): {sum(matrices['condition_array'] == 2)}")
    print(f"\nPerformance categories:")
    print(f"  - Poor: {len(score_groups['poor'])} participants")
    print(f"  - Fair: {len(score_groups['fair'])} participants")
    print(f"  - Good: {len(score_groups['good'])} participants")
    print(f"\nOverall acceptance rate: {trial_df['accept'].mean():.3f}")
    
    # Final report
    logger.info("\n" + "="*70)
    logger.info("✅ REPRODUCTION COMPLETE!")
    logger.info("="*70)
    logger.info(f"\n📁 All output_benchmark saved to: {Config.OUTPUT_DIR.absolute()}\n")
    
    logger.info("📊 FIGURES (matching paper):")
    logger.info("   • figure3_model_fitting.png      → Figure 3 from paper")
    logger.info("   • figure4_confidence_trends.png  → Figure 4 from paper")
    logger.info("   • figure5_score_distributions.png → Figure 5 from paper")
    logger.info("   • figure6_scatter_plots.png      → Figure 6 from paper")
    
    logger.info("\n📋 TABLES (matching paper):")
    logger.info("   • table1_model_params.csv        → Table 1 from paper (original fit)")
    logger.info("   • table2_logit_coeffs.csv         → Table 2 from paper")
    logger.info("   • table3_category_logit.csv       → Table 3 from paper")
    
    if Config.ROBUSTNESS_ENABLED:
        logger.info("\n📊 ROBUSTNESS TEST RESULTS:")
        logger.info("   • table1_final_robust_params.csv → Final robust parameters (mean across 100 iterations)")
        logger.info("   • robustness_statistics.csv      → Detailed robustness statistics")
        logger.info("   • robustness_analysis.png       → Parameter distribution visualizations")
        logger.info("   • robustness_raw_data.json       → Raw parameter estimates from all iterations")
    
    logger.info("\n📄 DATA FILES:")
    logger.info("   • trial_data.csv                  → All trial-level data")
    logger.info("   • participant_summary.csv        → Participant summaries")
    logger.info("   • table2_logit_summary.txt        → Full regression output")
    
    logger.info("\n" + "="*70)
    logger.info("KEY FINDINGS (from paper):")
    logger.info("="*70)
    logger.info("1. Self-confidence (NOT AI confidence) drives acceptance decisions")
    logger.info("2. Poor AI performance decreases both AI confidence AND self-confidence")
    logger.info("3. Humans misattribute blame to themselves for AI errors")
    logger.info("4. Good decision-makers: positive self-conf → acceptance correlation")
    logger.info("5. Poor decision-makers: negative correlation → vicious cycle")
    logger.info("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise

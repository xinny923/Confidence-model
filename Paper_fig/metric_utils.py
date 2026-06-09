from __future__ import annotations

from typing import Dict

import numpy as np


def condition_mean_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    condition_array: np.ndarray,
    num_trials: int,
    adjustment_k: int,
    adjustment_n: int = 99,
) -> Dict[str, float]:
    """Benchmark-style metrics on condition-level mean curves.

    Matches the truss benchmark's R2 denominator: each condition mean curve is
    centered by its own mean before SST values are summed. MSE is the mean
    squared error over the two condition curves and 30 formal trials.
    """
    obs_parts = []
    pred_parts = []
    sstot = 0.0

    for cond in sorted(np.unique(condition_array)):
        idx = np.where(condition_array == cond)[0]
        obs_mean = np.nanmean(observed[idx, 1:], axis=0)
        pred_mean = np.nanmean(predicted[idx, 1:], axis=0)
        obs_parts.append(obs_mean)
        pred_parts.append(pred_mean)
        sstot += float(np.nansum((obs_mean - np.nanmean(obs_mean)) ** 2))

    y = np.concatenate(obs_parts)
    y_hat = np.concatenate(pred_parts)
    ssres = float(np.nansum((y_hat - y) ** 2))
    n_points = int(np.count_nonzero(~np.isnan(y)))
    r2 = 1.0 - ssres / sstot if sstot > 0 else float("nan")
    # Match truss_og_benchmark.py / the MATLAB reporting convention used here:
    # adjusted R2 uses the fixed participant/parameter factor 99/92 even when
    # individual model variants expose a different number of free parameters.
    paper_adjustment = 99.0 / 92.0
    adj_r2 = 1.0 - paper_adjustment * (ssres / sstot) if sstot > 0 else float("nan")
    return {
        "mse": ssres / n_points if n_points > 0 else float("nan"),
        "r2": r2,
        "adj_r2": adj_r2,
        "ssres": ssres,
        "sstot": sstot,
        "n_metric_points": n_points,
    }

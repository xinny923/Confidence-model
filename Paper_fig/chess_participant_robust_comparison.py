from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

PAPER_DIR = Path(__file__).resolve().parent
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

import full_model as full  # noqa: E402
import og_model as og  # noqa: E402
from metric_utils import condition_mean_metrics  # noqa: E402


"""
Participant-level robust comparison for the manuscript aggregate table.

Model-family convention:
    - ORIG-BASE is Chong et al.'s 8-parameter baseline.
    - PHYS-BASE is the physics-skeleton rewrite of ORIG-BASE, so it still
      includes the explicit baseline/bias term alpha_b(B - C).
    - All incremental PHYS-BASE+... variants are reformulated models; in those
      rows B is absorbed into the model state and does not appear as a separate
      force term.
"""

OUT_DIR = PAPER_DIR / "aggregate_fit_output"
RNG_SEED = 42
N_ITER = 100
SUBSET_SIZE = 80


@dataclass(frozen=True)
class Variant:
    name: str
    evidence: str
    memory: str
    self_asym: bool
    align: bool
    explicit_b: bool = False


VARIANTS = (
    Variant("PHYS-BASE+MOD", "modify", "lagged", False, False),
    Variant("PHYS-BASE+MOD+A", "modify", "adaptive", False, False),
    Variant("PHYS-BASE+MOD+A+ASYM", "modify", "adaptive", True, False),
    Variant("PHYS-BASE+MOD+A+ALIGN", "modify", "adaptive", False, True),
    Variant("PHYS-FULL", "modify", "adaptive", True, True),
)

# The skeleton-change row keeps B explicit; every incremental reformulation
# above leaves explicit_b at its default False.
PHYS_BASE = Variant("PHYS-BASE", "og", "lagged", False, False, True)
PARAM_NAMES = [
    "alpha_fast",
    "alpha_slow",
    "alpha_b",
    "alpha_a",
    "gamma",
    "omega1",
    "omega2",
    "omega3",
    "omega4",
    "gamma_res",
]
INIT = {
    "aiconf": {
        "alpha_fast": 0.27,
        "alpha_slow": 0.35,
        "alpha_b": 0.04,
        "alpha_a": 0.34,
        "gamma": 0.30,
        "omega1": 0.84,
        "omega2": 0.21,
        "omega3": 0.0,
        "omega4": 0.52,
        "gamma_res": 0.0,
    },
    "selfconf": {
        "alpha_fast": 0.28,
        "alpha_slow": 0.47,
        "alpha_b": 0.12,
        "alpha_a": 0.11,
        "gamma": 0.28,
        "omega1": 0.57,
        "omega2": 0.83,
        "omega3": 0.24,
        "omega4": 0.29,
        "gamma_res": 0.1,
    },
}
BOUNDS = {
    "alpha_fast": (0.0, 2.0),
    "alpha_slow": (0.0, 2.0),
    "alpha_b": (0.0, 1.0),
    "alpha_a": (0.0, 1.0),
    "gamma": (0.0, 1.0),
    "omega1": (0.0, 1.0),
    "omega2": (0.0, 1.0),
    "omega3": (0.0, 1.0),
    "omega4": (0.0, 1.0),
    "gamma_res": (0.0, 10.0),
}


def names_for_variant(variant: Variant) -> list[str]:
    if variant.memory == "lagged":
        names = ["alpha_fast", "alpha_slow", "gamma", "omega1", "omega2", "omega3", "omega4"]
    else:
        names = ["alpha_fast", "alpha_slow", "alpha_a", "omega1", "omega2", "omega3", "omega4"]
    if variant.align:
        names.append("gamma_res")
    if variant.explicit_b:
        names.insert(2, "alpha_b")
    return names


def unpack(params: np.ndarray, names: list[str]) -> dict[str, float]:
    values = {name: 0.0 for name in PARAM_NAMES}
    values.update({name: float(value) for name, value in zip(names, params)})
    return values


def init_bounds(track: str, names: list[str]) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    init = np.array([INIT[track][name] for name in names], dtype=float)
    low = np.array([BOUNDS[name][0] for name in names], dtype=float)
    high = np.array([BOUNDS[name][1] for name in names], dtype=float)
    return init, (low, high)


def predict_variant(
    params: np.ndarray,
    names: list[str],
    variant: Variant,
    observed: np.ndarray,
    e_tensor: np.ndarray,
    align_tensor: np.ndarray,
    use_self_track: bool,
) -> np.ndarray:
    p = unpack(params, names)
    n_participants, series_len = observed.shape
    c = observed[:, 0].copy()
    b = observed[:, 0].copy()
    a = observed[:, 0].copy()
    pred = np.zeros_like(observed, dtype=float)
    pred[:, 0] = c

    for t in range(series_len - 1):
        if variant.memory == "lagged" and t > 0:
            a = p["gamma"] * pred[:, t - 1] + (1.0 - p["gamma"]) * a
        exp_term = (
            p["omega1"] * e_tensor[:, t, 0]
            + p["omega2"] * e_tensor[:, t, 1]
            + p["omega3"] * e_tensor[:, t, 2]
            + p["omega4"] * e_tensor[:, t, 3]
        )
        if variant.align:
            exp_term = exp_term * (1.0 + p["gamma_res"] * align_tensor[:, t])

        force = p["alpha_fast"] * (exp_term - c)
        if variant.explicit_b:
            force = force + p["alpha_b"] * (b - c)
        if use_self_track and variant.self_asym:
            force = force + p["alpha_slow"] * a
        else:
            force = force + p["alpha_slow"] * (a - c)
        c = np.clip(c + force, 0.0, 1.0)
        pred[:, t + 1] = c

        if variant.memory == "adaptive":
            a = a + p["alpha_a"] * (exp_term - a)

    return pred


def residuals_variant(
    params: np.ndarray,
    names: list[str],
    variant: Variant,
    observed: np.ndarray,
    e_tensor: np.ndarray,
    align_tensor: np.ndarray,
    use_self_track: bool,
) -> np.ndarray:
    pred = predict_variant(params, names, variant, observed, e_tensor, align_tensor, use_self_track)
    return (observed[:, 1:] - pred[:, 1:]).T.ravel()


def fit_variant(
    track: str,
    variant: Variant,
    observed: np.ndarray,
    e_tensor: np.ndarray,
    align_tensor: np.ndarray,
    condition_array: np.ndarray,
    use_self_track: bool,
) -> dict[str, Any]:
    names = names_for_variant(variant)
    init, bounds = init_bounds(track, names)
    res = least_squares(
        residuals_variant,
        init,
        bounds=bounds,
        args=(names, variant, observed, e_tensor, align_tensor, use_self_track),
        max_nfev=5000,
    )
    pred = predict_variant(res.x, names, variant, observed, e_tensor, align_tensor, use_self_track)
    metrics = condition_mean_metrics(observed, pred, condition_array, full.Config.NUM_TRIALS, adjustment_k=8)
    return {"params": res.x, "param_names": names, "pred": pred, "success": bool(res.success), **metrics}


def fit_og_track(track: str, observed: np.ndarray, e_tensor: np.ndarray, condition_array: np.ndarray) -> dict[str, Any]:
    init = np.array(og.Config.INITIAL_GUESS[track], dtype=float)
    res = least_squares(
        og.simulate_confidence_dynamics_extended,
        init,
        bounds=(np.zeros_like(init), np.ones_like(init)),
        args=(observed, e_tensor),
        max_nfev=5000,
    )
    pred = og.compute_model_predictions_extended(res.x, observed, e_tensor)
    metrics = condition_mean_metrics(observed, pred, condition_array, full.Config.NUM_TRIALS, adjustment_k=8)
    return {"params": res.x, "pred": pred, "success": bool(res.success), **metrics}


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in raw.groupby("Model", sort=False):
        row = {"Model": model}
        for metric in ["C_AI_MSE", "C_AI_adj_R2", "C_self_MSE", "C_self_adj_R2", "Mean_MSE", "Mean_adj_R2"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    og_parts, _ = og.load_all_participants(og.Config.DATA_DIR)
    og_mats = og.build_analysis_matrices(og_parts)
    full_parts, _ = full.load_participants(full.Config.DATA_DIR)
    full_mats = full.build_matrices(full_parts)

    rng = np.random.default_rng(RNG_SEED)
    n_participants = full_mats["ai_conf_matrix"].shape[0]
    rows = []
    param_rows = []

    for iteration in range(1, N_ITER + 1):
        idx = np.sort(rng.choice(n_participants, size=min(SUBSET_SIZE, n_participants), replace=False))
        condition_array = full_mats["condition_array"][idx]

        fit_rows: dict[str, dict[str, dict[str, Any]]] = {"ORIG-BASE": {}}
        for track, matrix_name in [("aiconf", "ai_conf_matrix"), ("selfconf", "self_conf_matrix")]:
            fit_rows["ORIG-BASE"][track] = fit_og_track(
                track,
                og_mats[matrix_name][idx],
                og_mats["e_tensor"][idx],
                og_mats["condition_array"][idx],
            )
            param_rows.append(
                {"iteration": iteration, "Model": "ORIG-BASE", "track": track, **dict(zip(og.Config.PARAMETER_NAMES, fit_rows["ORIG-BASE"][track]["params"]))}
            )
        fit_rows["PHYS-BASE"] = {}
        for track, matrix_name, use_self_track in [
            ("aiconf", "ai_conf_matrix", False),
            ("selfconf", "self_conf_matrix", True),
        ]:
            fit = fit_variant(
                track,
                PHYS_BASE,
                og_mats[matrix_name][idx],
                og_mats["e_tensor"][idx],
                np.zeros_like(full_mats["align_tensor"][idx]),
                og_mats["condition_array"][idx],
                use_self_track,
            )
            fit_rows["PHYS-BASE"][track] = fit
            param_rows.append(
                {
                    "iteration": iteration,
                    "Model": "PHYS-BASE",
                    "track": track,
                    **unpack(fit["params"], fit["param_names"]),
                }
            )

        for variant in VARIANTS:
            fit_rows[variant.name] = {}
            e_tensor = og_mats["e_tensor"][idx] if variant.evidence == "og" else full_mats["e_tensor"][idx]
            align_tensor = full_mats["align_tensor"][idx] if variant.align else np.zeros_like(full_mats["align_tensor"][idx])
            for track, matrix_name, use_self_track in [
                ("aiconf", "ai_conf_matrix", False),
                ("selfconf", "self_conf_matrix", True),
            ]:
                fit = fit_variant(
                    track,
                    variant,
                    full_mats[matrix_name][idx],
                    e_tensor,
                    align_tensor,
                    condition_array,
                    use_self_track,
                )
                fit_rows[variant.name][track] = fit
                param_rows.append(
                    {
                        "iteration": iteration,
                        "Model": variant.name,
                        "track": track,
                        **unpack(fit["params"], fit["param_names"]),
                    }
                )

        for model, fits in fit_rows.items():
            ai = fits["aiconf"]
            self_fit = fits["selfconf"]
            rows.append(
                {
                    "iteration": iteration,
                    "Model": model,
                    "C_AI_MSE": ai["mse"],
                    "C_AI_adj_R2": ai["adj_r2"],
                    "C_self_MSE": self_fit["mse"],
                    "C_self_adj_R2": self_fit["adj_r2"],
                    "Mean_MSE": (ai["mse"] + self_fit["mse"]) / 2.0,
                    "Mean_adj_R2": (ai["adj_r2"] + self_fit["adj_r2"]) / 2.0,
                }
            )

        if iteration % 10 == 0:
            print(f"robust iteration {iteration}/{N_ITER}")

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(OUT_DIR / "chess_participant_fit_robust_adj_raw.csv", index=False)
    summary.to_csv(OUT_DIR / "chess_participant_fit_robust_adj_summary.csv", index=False)
    pd.DataFrame(param_rows).to_csv(OUT_DIR / "chess_participant_fit_robust_params.csv", index=False)

    print("\nChess participant-level robust adjusted R2 summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()

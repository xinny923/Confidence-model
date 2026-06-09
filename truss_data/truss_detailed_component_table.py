"""
Detailed truss component table in the manuscript Table-3 model order.

Uses the complete reconstructed truss data with modify and alignment:
    ORIG-BASE
    PHYS-BASE
    MOD
    MOD+A
    MOD+A+ASYM
    MOD+A+ALIGN
    PHYS-FULL

Fitting follows the benchmark convention: participant-level residual fitting,
then condition-mean aggregate MSE/R2/adjusted R2 metrics.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import truss_detailed_subset_fit as detailed  # noqa: E402
from truss_og_benchmark import (  # noqa: E402
    Config as BaseConfig,
    INITIAL_GUESS as BASE_INITIAL,
    PARAMETER_NAMES as BASE_PARAM_NAMES,
    condition_mean_metric,
    fit_model as fit_baseline_model,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = ROOT / "data_folder_output"
OMEGA_NAMES = ["omega1", "omega2", "omega3", "omega4"]


@dataclass(frozen=True)
class Variant:
    model: str
    evidence: str
    memory: str
    self_asym: bool
    align: bool


VARIANTS: tuple[Variant, ...] = (
    Variant("MOD", "full", "lagged", False, False),
    Variant("MOD+A", "full", "adaptive", False, False),
    Variant("MOD+A+ASYM", "full", "adaptive", True, False),
    Variant("MOD+A+ALIGN", "full", "adaptive", False, True),
    Variant("PHYS-FULL", "full", "adaptive", True, True),
)


def param_names(variant: Variant) -> list[str]:
    if variant.memory == "lagged":
        names = ["alpha_fast", "alpha_slow", "gamma", *OMEGA_NAMES]
    else:
        names = ["alpha_fast", "alpha_slow", "alpha_a", *OMEGA_NAMES]
    if variant.align:
        names.append("gamma_res")
    return names


def initial_for(track: str, names: list[str]) -> np.ndarray:
    base = dict(zip(BASE_PARAM_NAMES, BASE_INITIAL[track]))
    full = {
        "aiconf": {
            "alpha_fast": 0.50,
            "alpha_slow": 0.50,
            "alpha_a": 0.50,
            "gamma": base["gamma"],
            "omega1": 0.84,
            "omega2": 0.21,
            "omega3": 0.50,
            "omega4": 0.00,
            "gamma_res": 0.00,
        },
        "selfconf": {
            "alpha_fast": 0.50,
            "alpha_slow": 0.50,
            "alpha_a": 0.50,
            "gamma": base["gamma"],
            "omega1": 0.57,
            "omega2": 0.83,
            "omega3": 0.50,
            "omega4": 0.00,
            "gamma_res": 0.10,
        },
    }[track]
    if "gamma" in names:
        full["alpha_fast"] = base["alpha_e"]
        full["alpha_slow"] = base["alpha_a"]
        for omega in OMEGA_NAMES:
            full[omega] = base[omega]
    return np.array([full[name] for name in names], dtype=float)


def bounds_for(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    high = np.array([10.0 if name == "gamma_res" else 1.0 for name in names], dtype=float)
    return np.zeros(len(names), dtype=float), high


def unpack(params: np.ndarray, names: list[str]) -> dict[str, float]:
    values = {name: 0.0 for name in ["alpha_fast", "alpha_slow", "alpha_a", "gamma", *OMEGA_NAMES, "gamma_res"]}
    values.update({name: float(value) for name, value in zip(names, params)})
    return values


def predict_variant(
    params: np.ndarray,
    names: list[str],
    variant: Variant,
    observed: np.ndarray,
    e_tensor: np.ndarray,
    align_matrix: np.ndarray,
    use_self_track: bool,
) -> np.ndarray:
    p = unpack(params, names)
    n_participants, series_len = observed.shape
    c = observed[:, 0].copy()
    a = observed[:, 0].copy()
    pred = np.zeros_like(observed, dtype=float)
    pred[:, 0] = c

    for t in range(series_len - 1):
        if variant.memory == "lagged" and t > 0:
            # Same lag convention as the OG benchmark: trial 2 still uses C0.
            a = p["gamma"] * pred[:, t - 1] + (1.0 - p["gamma"]) * a

        experience = (
            p["omega1"] * e_tensor[:, t, 0]
            + p["omega2"] * e_tensor[:, t, 1]
            + p["omega3"] * e_tensor[:, t, 2]
            + p["omega4"] * e_tensor[:, t, 3]
        )
        if variant.align:
            experience = experience * (1.0 + p["gamma_res"] * align_matrix[:, t])

        force = p["alpha_fast"] * (experience - c)
        if use_self_track and variant.self_asym:
            force = force + p["alpha_slow"] * a
        else:
            force = force + p["alpha_slow"] * (a - c)

        c = np.clip(c + force, 0.0, 1.0)
        pred[:, t + 1] = c

        if variant.memory == "adaptive":
            a = a + p["alpha_a"] * (experience - a)

    return pred


def residuals_variant(
    params: np.ndarray,
    names: list[str],
    variant: Variant,
    observed: np.ndarray,
    e_tensor: np.ndarray,
    align_matrix: np.ndarray,
    use_self_track: bool,
) -> np.ndarray:
    pred = predict_variant(params, names, variant, observed, e_tensor, align_matrix, use_self_track)
    return (observed[:, 1:] - pred[:, 1:]).T.ravel()


def fit_variant_track(
    variant: Variant,
    track: str,
    observed: np.ndarray,
    e_tensor: np.ndarray,
    align_matrix: np.ndarray,
    condition_array: np.ndarray,
    use_self_track: bool,
) -> dict[str, Any]:
    names = param_names(variant)
    result = least_squares(
        residuals_variant,
        x0=initial_for(track, names),
        bounds=bounds_for(names),
        args=(names, variant, observed, e_tensor, align_matrix, use_self_track),
        max_nfev=BaseConfig.max_nfev,
    )
    pred = predict_variant(result.x, names, variant, observed, e_tensor, align_matrix, use_self_track)
    metrics = condition_mean_metric(observed, pred, condition_array, len(names))
    return {
        "params": result.x,
        "param_names": names,
        "pred": pred,
        "success": bool(result.success),
        "nfev": int(result.nfev),
        **metrics,
    }


def summarize_fit(model_label: str, ai_fit: dict[str, Any], self_fit: dict[str, Any]) -> dict[str, float | str]:
    return {
        "Model": model_label,
        "C_AI_MSE": ai_fit["mse"],
        "C_AI_R2": ai_fit["r2"],
        "C_AI_adj_R2": ai_fit["adj_r2"],
        "C_self_MSE": self_fit["mse"],
        "C_self_R2": self_fit["r2"],
        "C_self_adj_R2": self_fit["adj_r2"],
        "Mean_MSE": (ai_fit["mse"] + self_fit["mse"]) / 2.0,
        "Mean_R2": (ai_fit["r2"] + self_fit["r2"]) / 2.0,
        "Mean_adj_R2": (ai_fit["adj_r2"] + self_fit["adj_r2"]) / 2.0,
    }


def fit_component_table(matrices: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    param_rows = []
    zero_align = np.zeros_like(matrices["align_matrix"])

    base_ai = fit_baseline_model("aiconf", matrices["ai_conf_matrix"], matrices["baseline_e_tensor"], matrices["condition_array"])
    base_self = fit_baseline_model("selfconf", matrices["self_conf_matrix"], matrices["baseline_e_tensor"], matrices["condition_array"])

    for label in ["ORIG-BASE", "PHYS-BASE"]:
        rows.append(summarize_fit(label, base_ai, base_self))
        for track, fit in [("aiconf", base_ai), ("selfconf", base_self)]:
            row = {"Model": label, "track": track, "success": fit["success"], "nfev": fit["nfev"]}
            row.update(dict(zip(BASE_PARAM_NAMES, fit["params"])))
            param_rows.append(row)

    for variant in VARIANTS:
        align = matrices["align_matrix"] if variant.align else zero_align
        ai_fit = fit_variant_track(
            variant,
            "aiconf",
            matrices["ai_conf_matrix"],
            matrices["full_e_tensor"],
            align,
            matrices["condition_array"],
            False,
        )
        self_fit = fit_variant_track(
            variant,
            "selfconf",
            matrices["self_conf_matrix"],
            matrices["full_e_tensor"],
            align,
            matrices["condition_array"],
            True,
        )
        rows.append(summarize_fit(variant.model, ai_fit, self_fit))
        for track, fit in [("aiconf", ai_fit), ("selfconf", self_fit)]:
            row = {"Model": variant.model, "track": track, "success": fit["success"], "nfev": fit["nfev"]}
            row.update(dict(zip(fit["param_names"], fit["params"])))
            param_rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(param_rows)


def subset_matrices(matrices: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ai_conf_matrix": matrices["ai_conf_matrix"][idx],
        "self_conf_matrix": matrices["self_conf_matrix"][idx],
        "baseline_e_tensor": matrices["baseline_e_tensor"][idx],
        "full_e_tensor": matrices["full_e_tensor"][idx],
        "align_matrix": matrices["align_matrix"][idx],
        "condition_array": matrices["condition_array"][idx],
    }


def run_robustness(
    matrices: dict[str, np.ndarray],
    n_iter: int = BaseConfig.robustness_iterations,
    subset_per_condition: int = BaseConfig.robustness_subset_per_condition,
) -> pd.DataFrame:
    rng = np.random.default_rng(BaseConfig.random_seed)
    rows = []
    by_condition = {
        cond: np.where(matrices["condition_array"] == cond)[0]
        for cond in sorted(np.unique(matrices["condition_array"]))
    }
    for iteration in range(1, n_iter + 1):
        sampled = []
        for idx in by_condition.values():
            sampled.extend(rng.choice(idx, size=min(subset_per_condition, len(idx)), replace=False).tolist())
        sub = subset_matrices(matrices, np.array(sorted(sampled), dtype=int))
        table, _ = fit_component_table(sub)
        for row in table.to_dict("records"):
            row["iteration"] = iteration
            rows.append(row)
        if iteration % 10 == 0:
            logger.info("Detailed component robustness iteration %d/%d", iteration, n_iter)
    return pd.DataFrame(rows)


def summarize_robustness(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["C_AI_MSE", "C_AI_adj_R2", "C_self_MSE", "C_self_adj_R2", "Mean_MSE", "Mean_adj_R2"]
    rows = []
    for model, group in raw.groupby("Model", sort=False):
        row = {"Model": model}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrices, metadata = detailed.build_subset_matrices()
    logger.info(
        "Fitting detailed component table for %d participants; fallback=%d",
        matrices["ai_conf_matrix"].shape[0],
        int(matrices["fallback_trials"][0]),
    )
    table, params = fit_component_table(matrices)
    raw = run_robustness(matrices)
    summary = summarize_robustness(raw)

    metadata.to_csv(OUT_DIR / "detailed_component_metadata.csv", index=False)
    table.to_csv(OUT_DIR / "detailed_component_metrics.csv", index=False)
    params.to_csv(OUT_DIR / "detailed_component_params.csv", index=False)
    raw.to_csv(OUT_DIR / "detailed_component_robust_raw.csv", index=False)
    summary.to_csv(OUT_DIR / "detailed_component_robust_summary.csv", index=False)

    print("\nDetailed truss component robust summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved: {OUT_DIR / 'detailed_component_robust_summary.csv'}")


if __name__ == "__main__":
    main()

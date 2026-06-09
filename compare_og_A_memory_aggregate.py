from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
PAPER_DIR = ROOT / "Paper_fig"
TRUSS_DIR = ROOT / "truss_data"
for path in (PAPER_DIR, TRUSS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import chess_participant_robust_comparison as chess  # noqa: E402
import full_model as chess_full  # noqa: E402
import og_model as chess_og  # noqa: E402
from metric_utils import condition_mean_metrics  # noqa: E402

import truss_detailed_component_table as truss  # noqa: E402
import truss_detailed_subset_fit as truss_detailed  # noqa: E402
from truss_og_benchmark import Config as TrussConfig  # noqa: E402
from truss_og_benchmark import condition_mean_metric  # noqa: E402
from truss_og_benchmark import fit_model as fit_truss_baseline  # noqa: E402


@dataclass(frozen=True)
class Variant:
    model: str
    evidence: str
    memory: str
    self_asym: bool
    align: bool
    explicit_b: bool = False


CHESS_VARIANTS = (
    Variant("MOD", "modify", "og_history", False, False),
    Variant("MOD+A_OG", "modify", "og_history", False, False),
    Variant("MOD+A_OG+ASYM", "modify", "og_history", True, False),
    Variant("MOD+A_OG+ALIGN", "modify", "og_history", False, True),
    Variant("PHYS-FULL_OG-A", "modify", "og_history", True, True),
)

TRUSS_VARIANTS = (
    Variant("MOD", "full", "og_history", False, False),
    Variant("MOD+A_OG", "full", "og_history", False, False),
    Variant("MOD+A_OG+ASYM", "full", "og_history", True, False),
    Variant("MOD+A_OG+ALIGN", "full", "og_history", False, True),
    Variant("PHYS-FULL_OG-A", "full", "og_history", True, True),
)


def names_for_variant(variant: Variant, include_b: bool = False) -> list[str]:
    names = ["alpha_fast", "alpha_slow", "gamma", "omega1", "omega2", "omega3", "omega4"]
    if variant.align:
        names.append("gamma_res")
    if include_b or variant.explicit_b:
        names.insert(2, "alpha_b")
    return names


def unpack(params: np.ndarray, names: list[str]) -> dict[str, float]:
    values = {name: 0.0 for name in ["alpha_fast", "alpha_slow", "alpha_b", "gamma", "omega1", "omega2", "omega3", "omega4", "gamma_res"]}
    values.update({name: float(value) for name, value in zip(names, params)})
    return values


def init_bounds_chess(track: str, names: list[str]) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    init = []
    low = []
    high = []
    for name in names:
        init.append(chess.INIT[track][name])
        low.append(chess.BOUNDS[name][0])
        high.append(chess.BOUNDS[name][1])
    return np.array(init, dtype=float), (np.array(low, dtype=float), np.array(high, dtype=float))


def init_bounds_truss(track: str, names: list[str]) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    init = truss.initial_for(track, names)
    low, high = truss.bounds_for(names)
    return init, (low, high)


def predict_og_history(
    params: np.ndarray,
    names: list[str],
    variant: Variant,
    observed: np.ndarray,
    e_tensor: np.ndarray,
    align_matrix: np.ndarray,
    use_self_track: bool,
) -> np.ndarray:
    p = unpack(params, names)
    c = observed[:, 0].copy()
    a = observed[:, 0].copy()
    b = observed[:, 0].copy()
    pred = np.zeros_like(observed, dtype=float)
    pred[:, 0] = c

    for t in range(observed.shape[1] - 1):
        if t > 0:
            a = p["gamma"] * pred[:, t - 1] + (1.0 - p["gamma"]) * a

        exp_term = (
            p["omega1"] * e_tensor[:, t, 0]
            + p["omega2"] * e_tensor[:, t, 1]
            + p["omega3"] * e_tensor[:, t, 2]
            + p["omega4"] * e_tensor[:, t, 3]
        )
        if variant.align:
            exp_term = exp_term * (1.0 + p["gamma_res"] * align_matrix[:, t])

        force = p["alpha_fast"] * (exp_term - c)
        if variant.explicit_b:
            force = force + p["alpha_b"] * (b - c)
        if use_self_track and variant.self_asym:
            force = force + p["alpha_slow"] * a
        else:
            force = force + p["alpha_slow"] * (a - c)

        c = np.clip(c + force, 0.0, 1.0)
        pred[:, t + 1] = c

    return pred


def residuals_og_history(params, names, variant, observed, e_tensor, align_matrix, use_self_track):
    pred = predict_og_history(params, names, variant, observed, e_tensor, align_matrix, use_self_track)
    return (observed[:, 1:] - pred[:, 1:]).T.ravel()


def fit_chess_variant(track: str, variant: Variant, observed: np.ndarray, e_tensor: np.ndarray, align_tensor: np.ndarray, condition_array: np.ndarray, use_self_track: bool) -> dict[str, Any]:
    names = names_for_variant(variant)
    init, bounds = init_bounds_chess(track, names)
    res = least_squares(
        residuals_og_history,
        init,
        bounds=bounds,
        args=(names, variant, observed, e_tensor, align_tensor, use_self_track),
        max_nfev=5000,
    )
    pred = predict_og_history(res.x, names, variant, observed, e_tensor, align_tensor, use_self_track)
    metrics = condition_mean_metrics(observed, pred, condition_array, chess_full.Config.NUM_TRIALS, adjustment_k=8)
    return {"params": res.x, "param_names": names, "success": bool(res.success), **metrics}


def fit_truss_variant(variant: Variant, track: str, observed: np.ndarray, e_tensor: np.ndarray, align_matrix: np.ndarray, condition_array: np.ndarray, use_self_track: bool) -> dict[str, Any]:
    names = names_for_variant(variant)
    init, bounds = init_bounds_truss(track, names)
    res = least_squares(
        residuals_og_history,
        init,
        bounds=bounds,
        args=(names, variant, observed, e_tensor, align_matrix, use_self_track),
        max_nfev=TrussConfig.max_nfev,
    )
    pred = predict_og_history(res.x, names, variant, observed, e_tensor, align_matrix, use_self_track)
    metrics = condition_mean_metric(observed, pred, condition_array, len(names))
    return {"params": res.x, "param_names": names, "success": bool(res.success), "nfev": int(res.nfev), **metrics}


def add_summary_row(rows: list[dict[str, float | str | int]], iteration: int, model: str, ai: dict[str, Any], self_fit: dict[str, Any]) -> None:
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


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in raw.groupby("Model", sort=False):
        row = {"Model": model}
        for metric in ["C_AI_MSE", "C_AI_adj_R2", "C_self_MSE", "C_self_adj_R2", "Mean_MSE", "Mean_adj_R2"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def run_chess() -> pd.DataFrame:
    og_parts, _ = chess_og.load_all_participants(chess_og.Config.DATA_DIR)
    og_mats = chess_og.build_analysis_matrices(og_parts)
    full_parts, _ = chess_full.load_participants(chess_full.Config.DATA_DIR)
    full_mats = chess_full.build_matrices(full_parts)
    rng = np.random.default_rng(chess.RNG_SEED)
    rows: list[dict[str, float | str | int]] = []
    n_participants = full_mats["ai_conf_matrix"].shape[0]

    for iteration in range(1, chess.N_ITER + 1):
        idx = np.sort(rng.choice(n_participants, size=min(chess.SUBSET_SIZE, n_participants), replace=False))
        for variant in CHESS_VARIANTS:
            e_tensor = full_mats["e_tensor"][idx]
            align_tensor = full_mats["align_tensor"][idx] if variant.align else np.zeros_like(full_mats["align_tensor"][idx])
            ai = fit_chess_variant("aiconf", variant, full_mats["ai_conf_matrix"][idx], e_tensor, align_tensor, full_mats["condition_array"][idx], False)
            self_fit = fit_chess_variant("selfconf", variant, full_mats["self_conf_matrix"][idx], e_tensor, align_tensor, full_mats["condition_array"][idx], True)
            add_summary_row(rows, iteration, variant.model, ai, self_fit)
        if iteration % 10 == 0:
            print(f"chess OG-A iteration {iteration}/{chess.N_ITER}")

    raw = pd.DataFrame(rows)
    out = PAPER_DIR / "aggregate_fit_output"
    raw.to_csv(out / "chess_og_A_memory_robust_raw.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(out / "chess_og_A_memory_robust_summary.csv", index=False)
    return summary


def subset_truss(matrices: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ai_conf_matrix": matrices["ai_conf_matrix"][idx],
        "self_conf_matrix": matrices["self_conf_matrix"][idx],
        "baseline_e_tensor": matrices["baseline_e_tensor"][idx],
        "full_e_tensor": matrices["full_e_tensor"][idx],
        "align_matrix": matrices["align_matrix"][idx],
        "condition_array": matrices["condition_array"][idx],
    }


def run_truss() -> pd.DataFrame:
    matrices, _ = truss_detailed.build_subset_matrices()
    rng = np.random.default_rng(TrussConfig.random_seed)
    by_condition = {
        cond: np.where(matrices["condition_array"] == cond)[0]
        for cond in sorted(np.unique(matrices["condition_array"]))
    }
    rows: list[dict[str, float | str | int]] = []

    for iteration in range(1, TrussConfig.robustness_iterations + 1):
        sampled = []
        for idx in by_condition.values():
            sampled.extend(rng.choice(idx, size=min(TrussConfig.robustness_subset_per_condition, len(idx)), replace=False).tolist())
        sub = subset_truss(matrices, np.array(sorted(sampled), dtype=int))
        for variant in TRUSS_VARIANTS:
            align = sub["align_matrix"] if variant.align else np.zeros_like(sub["align_matrix"])
            ai = fit_truss_variant(variant, "aiconf", sub["ai_conf_matrix"], sub["full_e_tensor"], align, sub["condition_array"], False)
            self_fit = fit_truss_variant(variant, "selfconf", sub["self_conf_matrix"], sub["full_e_tensor"], align, sub["condition_array"], True)
            add_summary_row(rows, iteration, variant.model, ai, self_fit)
        if iteration % 10 == 0:
            print(f"truss OG-A iteration {iteration}/{TrussConfig.robustness_iterations}")

    raw = pd.DataFrame(rows)
    out = TRUSS_DIR / "data_folder_output"
    raw.to_csv(out / "detailed_og_A_memory_robust_raw.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(out / "detailed_og_A_memory_robust_summary.csv", index=False)
    return summary


def main() -> None:
    chess_summary = run_chess()
    truss_summary = run_truss()
    print("\nChess OG-style A robust summary")
    print(chess_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTruss OG-style A robust summary")
    print(truss_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()

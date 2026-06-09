"""
Fit baseline and partial-full confidence models on the currently available
detailed truss participants.

This uses the extracted `truss_data/data/.P#/data#.csv` files for confidence,
the reconstructed detailed action table for modify/alignment, and the same
condition-mean robust adjusted R2 convention used by truss_og_benchmark.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truss_og_benchmark import (  # noqa: E402
    Config as BaseConfig,
    INITIAL_GUESS,
    PARAMETER_NAMES as BASE_PARAM_NAMES,
    _cell_to_conditions,
    _load_first_variable,
    condition_mean_metric,
    fit_model as fit_baseline_model,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DATA_DIR = ROOT / "data"
RECON_PATH = ROOT / "data_folder_output" / "data_folder_reconstructed_trials.csv"
OUTPUT_DIR = ROOT / "data_folder_output"
NUM_TRIALS = 30
FULL_PARAM_NAMES = [
    "alpha_fast",
    "alpha_slow",
    "alpha_a",
    "omega1",
    "omega2",
    "omega3",
    "omega4",
    "gamma_res",
]
FULL_INITIAL = {
    "aiconf": np.array([0.50, 0.50, 0.50, 0.84, 0.21, 0.50, 0.00, 0.00], dtype=float),
    "selfconf": np.array([0.50, 0.50, 0.50, 0.57, 0.83, 0.50, 0.00, 0.10], dtype=float),
}
FULL_BOUNDS = (
    np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
    np.array([1, 1, 1, 1, 1, 1, 1, 10], dtype=float),
)
MODIFY_SPLIT_AI = 0.5
MODIFY_SPLIT_SELF = 0.5


def load_original_e_by_condition() -> Dict[int, np.ndarray]:
    return _cell_to_conditions(_load_first_variable(ROOT / "e_data.mat"), 3, "e_data")


def read_confidence_csv(pid: int) -> Tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / f".P{pid}" / f"data{pid}.csv"
    values = pd.read_csv(path, header=None).to_numpy(dtype=float)
    ai = values[2:, 0]
    self_conf = values[2:, 1]
    if ai.shape[0] != NUM_TRIALS + 1 or self_conf.shape[0] != NUM_TRIALS + 1:
        raise ValueError(f"{path} should provide {NUM_TRIALS + 1} modeled confidence points, got {ai.shape[0]}")
    return ai, self_conf


def write_evidence(e: np.ndarray, trial_idx: int, first_pair: bool, accept_weight: float, reject_weight: float) -> None:
    if first_pair:
        e[trial_idx, 0] = accept_weight
        e[trial_idx, 1] = reject_weight
    else:
        e[trial_idx, 2] = accept_weight
        e[trial_idx, 3] = reject_weight


def build_subset_matrices() -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    recon = pd.read_csv(RECON_PATH)
    original_e_by_cond = load_original_e_by_condition()

    participants = (
        recon[["participant", "condition", "condition_row"]]
        .drop_duplicates()
        .sort_values(["condition", "condition_row", "participant"])
    )

    ai_conf = []
    self_conf = []
    baseline_e = []
    full_e = []
    align = []
    conditions = []
    metadata_rows = []
    fallback_trials = 0

    for row in participants.itertuples(index=False):
        pid = int(row.participant)
        cond = int(row.condition)
        condition_row = int(row.condition_row)
        ai_series, self_series = read_confidence_csv(pid)

        person = recon[(recon["participant"] == pid) & (recon["condition"] == cond)].sort_values("trial")
        if person.shape[0] != NUM_TRIALS:
            raise ValueError(f"P{pid} condition {cond} has {person.shape[0]} reconstructed rows")

        original_e = original_e_by_cond[cond][condition_row, :, :].astype(float)
        e_base = original_e.copy()
        e_full = np.zeros((NUM_TRIALS, 4), dtype=float)
        align_series = np.zeros(NUM_TRIALS, dtype=float)

        for rec in person.itertuples(index=False):
            t = int(rec.trial) - 1
            original_channel = int(np.nanargmax(original_e[t]))
            first_pair = original_channel in (0, 1)
            action = str(rec.reconstructed_action)
            has_detail = bool(rec.has_move1) and bool(rec.has_move2) and bool(rec.has_suggestion) and action != "missing"
            if not has_detail:
                fallback_trials += 1
                e_full[t, :] = original_e[t, :]
                align_series[t] = 0.0
                continue

            if action == "accept_ai":
                write_evidence(e_full, t, first_pair, 1.0, 0.0)
            elif action == "reject_ai":
                write_evidence(e_full, t, first_pair, 0.0, 1.0)
            elif action == "modify":
                write_evidence(e_full, t, first_pair, MODIFY_SPLIT_AI, MODIFY_SPLIT_SELF)
            else:
                raise ValueError(f"Unexpected action {action!r} for P{pid} trial {t + 1}")

            align_series[t] = float(rec.align) if has_detail and not pd.isna(rec.align) else 0.0

        ai_conf.append(ai_series)
        self_conf.append(self_series)
        baseline_e.append(e_base)
        full_e.append(e_full)
        align.append(align_series)
        conditions.append(cond)
        metadata_rows.append(
            {
                "participant": pid,
                "condition": cond,
                "condition_row": condition_row,
                "available_detailed_trials": int(
                    (
                        person["has_move1"]
                        & person["has_move2"]
                        & person["has_suggestion"]
                        & (person["reconstructed_action"] != "missing")
                    ).sum()
                ),
            }
        )

    matrices = {
        "ai_conf_matrix": np.vstack(ai_conf),
        "self_conf_matrix": np.vstack(self_conf),
        "baseline_e_tensor": np.stack(baseline_e),
        "full_e_tensor": np.stack(full_e),
        "align_matrix": np.vstack(align),
        "condition_array": np.asarray(conditions, dtype=int),
        "participant_array": participants["participant"].to_numpy(dtype=int),
        "fallback_trials": np.asarray([fallback_trials], dtype=int),
    }
    return matrices, pd.DataFrame(metadata_rows)


def predict_full(params: np.ndarray, observed: np.ndarray, e_tensor: np.ndarray, align: np.ndarray, use_self_track: bool) -> np.ndarray:
    alpha_fast, alpha_slow, alpha_a, omega1, omega2, omega3, omega4, gamma_res = params
    n_participants, series_len = observed.shape
    n_trials = series_len - 1
    c = observed[:, 0].copy()
    a = observed[:, 0].copy()
    predicted = np.zeros_like(observed, dtype=float)
    predicted[:, 0] = c

    for t in range(n_trials):
        experience = (
            omega1 * e_tensor[:, t, 0]
            + omega2 * e_tensor[:, t, 1]
            + omega3 * e_tensor[:, t, 2]
            + omega4 * e_tensor[:, t, 3]
        )
        experience = experience * (1.0 + gamma_res * align[:, t])

        force = alpha_fast * (experience - c)
        if use_self_track:
            force = force + alpha_slow * a
        else:
            force = force + alpha_slow * (a - c)

        c = np.clip(c + force, 0.0, 1.0)
        predicted[:, t + 1] = c
        a = a + alpha_a * (experience - a)

    return predicted


def residuals_full(params: np.ndarray, observed: np.ndarray, e_tensor: np.ndarray, align: np.ndarray, use_self_track: bool) -> np.ndarray:
    predicted = predict_full(params, observed, e_tensor, align, use_self_track)
    return (observed[:, 1:] - predicted[:, 1:]).T.ravel()


def fit_full_model(model_name: str, observed: np.ndarray, e_tensor: np.ndarray, align: np.ndarray, condition_array: np.ndarray, use_self_track: bool) -> Dict[str, Any]:
    result = least_squares(
        residuals_full,
        x0=FULL_INITIAL[model_name],
        bounds=FULL_BOUNDS,
        args=(observed, e_tensor, align, use_self_track),
        max_nfev=BaseConfig.max_nfev,
    )
    predicted = predict_full(result.x, observed, e_tensor, align, use_self_track)
    metrics = condition_mean_metric(observed, predicted, condition_array, len(FULL_PARAM_NAMES))
    metrics.update({"params": result.x, "nfev": int(result.nfev), "success": bool(result.success)})
    return metrics


def summarize_fit(model_label: str, ai_fit: Dict[str, Any], self_fit: Dict[str, Any]) -> Dict[str, float | str]:
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


def fit_variant_pair(
    model_label: str,
    matrices: Dict[str, np.ndarray],
    e_tensor: np.ndarray,
    align_tensor: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ai_fit = fit_full_model(
        "aiconf",
        matrices["ai_conf_matrix"],
        e_tensor,
        align_tensor,
        matrices["condition_array"],
        use_self_track=False,
    )
    self_fit = fit_full_model(
        "selfconf",
        matrices["self_conf_matrix"],
        e_tensor,
        align_tensor,
        matrices["condition_array"],
        use_self_track=True,
    )
    return ai_fit, self_fit


def subset_matrices(matrices: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "ai_conf_matrix": matrices["ai_conf_matrix"][idx],
        "self_conf_matrix": matrices["self_conf_matrix"][idx],
        "baseline_e_tensor": matrices["baseline_e_tensor"][idx],
        "full_e_tensor": matrices["full_e_tensor"][idx],
        "align_matrix": matrices["align_matrix"][idx],
        "condition_array": matrices["condition_array"][idx],
        "participant_array": matrices["participant_array"][idx],
    }


def detailed_variant_specs(matrices: Dict[str, np.ndarray]) -> list[tuple[str, str]]:
    return [
        ("PHYS-FULL-NO-MOD-NO-ALIGN", "baseline_no_align"),
        ("PHYS-FULL+MOD", "full_no_align"),
        ("PHYS-FULL+ALIGN", "baseline_align"),
        ("PHYS-FULL-DETAILED", "full_align"),
    ]


def tensors_for_variant(matrices: Dict[str, np.ndarray], spec: str) -> Tuple[np.ndarray, np.ndarray]:
    zero_align = np.zeros_like(matrices["align_matrix"])
    if spec == "baseline_no_align":
        return matrices["baseline_e_tensor"], zero_align
    if spec == "full_no_align":
        return matrices["full_e_tensor"], zero_align
    if spec == "baseline_align":
        return matrices["baseline_e_tensor"], matrices["align_matrix"]
    if spec == "full_align":
        return matrices["full_e_tensor"], matrices["align_matrix"]
    raise ValueError(f"Unknown detailed variant spec {spec}")


def run_detailed_robustness(
    matrices: Dict[str, np.ndarray],
    n_iter: int = BaseConfig.robustness_iterations,
    subset_per_condition: int = BaseConfig.robustness_subset_per_condition,
) -> pd.DataFrame:
    rng = np.random.default_rng(BaseConfig.random_seed)
    rows = []
    conditions = np.unique(matrices["condition_array"])
    for iteration in range(1, n_iter + 1):
        sampled = []
        for cond in conditions:
            cond_idx = np.where(matrices["condition_array"] == cond)[0]
            if cond_idx.size < subset_per_condition:
                raise ValueError(f"Condition {cond} has {cond_idx.size} participants, need {subset_per_condition}")
            sampled.extend(rng.choice(cond_idx, size=subset_per_condition, replace=False).tolist())
        idx = np.array(sampled, dtype=int)
        sub = subset_matrices(matrices, idx)

        base_ai = fit_baseline_model("aiconf", sub["ai_conf_matrix"], sub["baseline_e_tensor"], sub["condition_array"])
        base_self = fit_baseline_model("selfconf", sub["self_conf_matrix"], sub["baseline_e_tensor"], sub["condition_array"])
        base_row = summarize_fit("BASELINE-OG", base_ai, base_self)
        base_row.update({f"AI_{name}": value for name, value in zip(BASE_PARAM_NAMES, base_ai["params"])})
        base_row.update({f"self_{name}": value for name, value in zip(BASE_PARAM_NAMES, base_self["params"])})
        rows.append({
            "iteration": iteration,
            **base_row,
        })

        for model_label, spec in detailed_variant_specs(sub):
            e_tensor, align_tensor = tensors_for_variant(sub, spec)
            ai_fit, self_fit = fit_variant_pair(model_label, sub, e_tensor, align_tensor)
            row = summarize_fit(model_label, ai_fit, self_fit)
            row.update({f"AI_{name}": value for name, value in zip(FULL_PARAM_NAMES, ai_fit["params"])})
            row.update({f"self_{name}": value for name, value in zip(FULL_PARAM_NAMES, self_fit["params"])})
            rows.append({
                "iteration": iteration,
                **row,
            })
        if iteration % 10 == 0:
            logger.info("Detailed robustness iteration %d/%d", iteration, n_iter)
    return pd.DataFrame(rows)


def summarize_detailed_robustness(robust: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["C_AI_MSE", "C_AI_adj_R2", "C_self_MSE", "C_self_adj_R2", "Mean_MSE", "Mean_adj_R2"]
    for model, group in robust.groupby("Model", sort=False):
        row = {"Model": model}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrices, metadata = build_subset_matrices()
    logger.info(
        "Fitting %d participants; fallback detailed trials=%d",
        matrices["ai_conf_matrix"].shape[0],
        int(matrices["fallback_trials"][0]),
    )

    base_ai = fit_baseline_model(
        "aiconf",
        matrices["ai_conf_matrix"],
        matrices["baseline_e_tensor"],
        matrices["condition_array"],
    )
    base_self = fit_baseline_model(
        "selfconf",
        matrices["self_conf_matrix"],
        matrices["baseline_e_tensor"],
        matrices["condition_array"],
    )
    zero_align = np.zeros_like(matrices["align_matrix"])
    full_variants = [
        (model_label, *tensors_for_variant(matrices, spec))
        for model_label, spec in detailed_variant_specs(matrices)
    ]
    variant_fits = []
    for model_label, e_tensor, align_tensor in full_variants:
        ai_fit, self_fit = fit_variant_pair(model_label, matrices, e_tensor, align_tensor)
        variant_fits.append((model_label, ai_fit, self_fit))

    table = pd.DataFrame([summarize_fit("BASELINE-OG", base_ai, base_self)] + [
        summarize_fit(model_label, ai_fit, self_fit)
        for model_label, ai_fit, self_fit in variant_fits
    ])
    params = []
    for label, track, names, fit in [
        ("BASELINE-OG", "aiconf", BASE_PARAM_NAMES, base_ai),
        ("BASELINE-OG", "selfconf", BASE_PARAM_NAMES, base_self),
        *[
            (model_label, track, FULL_PARAM_NAMES, fit)
            for model_label, ai_fit, self_fit in variant_fits
            for track, fit in [("aiconf", ai_fit), ("selfconf", self_fit)]
        ],
    ]:
        row = {"Model": label, "track": track, "success": fit["success"], "nfev": fit["nfev"]}
        row.update(dict(zip(names, fit["params"])))
        params.append(row)

    metadata.to_csv(OUTPUT_DIR / "detailed_available_metadata.csv", index=False)
    table.to_csv(OUTPUT_DIR / "detailed_available_baseline_full_metrics.csv", index=False)
    pd.DataFrame(params).to_csv(OUTPUT_DIR / "detailed_available_baseline_full_params.csv", index=False)

    print("\nDetailed truss available data: baseline vs full")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nParticipants: {matrices['ai_conf_matrix'].shape[0]}")
    print(f"Detailed-trial fallback count: {int(matrices['fallback_trials'][0])} / {matrices['ai_conf_matrix'].shape[0] * NUM_TRIALS}")
    print(f"Saved: {OUTPUT_DIR / 'detailed_available_baseline_full_metrics.csv'}")

    robust = run_detailed_robustness(matrices)
    robust_summary = summarize_detailed_robustness(robust)
    robust.to_csv(OUTPUT_DIR / "detailed_available_robust_raw.csv", index=False)
    robust_summary.to_csv(OUTPUT_DIR / "detailed_available_robust_summary.csv", index=False)
    print("\nDetailed truss robust summary")
    print(robust_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved: {OUTPUT_DIR / 'detailed_available_robust_summary.csv'}")


if __name__ == "__main__":
    main()

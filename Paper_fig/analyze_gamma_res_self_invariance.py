from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "Paper_fig"
TRUSS_DIR = ROOT / "truss_data"
OUT_DIR = PAPER_DIR / "full_output" / "gamma_res_self_invariance"

if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))
if str(TRUSS_DIR) not in sys.path:
    sys.path.insert(0, str(TRUSS_DIR))

import full_model as chess  # noqa: E402
from truss_detailed_subset_fit import build_subset_matrices as build_truss_matrices  # noqa: E402
from truss_detailed_subset_fit import fit_full_model as fit_truss_full_model  # noqa: E402
from truss_detailed_subset_fit import tensors_for_variant  # noqa: E402


PARAMS = chess.Config.PARAMETER_NAMES
GAMMA_IDX = PARAMS.index("gamma_res")
OMEGA3_IDX = PARAMS.index("omega3")
BETA_IDX = PARAMS.index("beta")


def ai_bounds_and_init() -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    init = np.array(chess.Config.INITIAL_GUESS["aiconf"], dtype=float)
    low = chess.Config.BOUNDS_LOW_DEFAULT.copy()
    high = chess.Config.BOUNDS_HIGH_DEFAULT.copy()
    init[OMEGA3_IDX] = 0.0
    low[OMEGA3_IDX] = 0.0
    high[OMEGA3_IDX] = 1e-12
    init[BETA_IDX] = chess.Config.BETA_FIXED
    low[BETA_IDX] = chess.Config.BETA_FIXED
    high[BETA_IDX] = chess.Config.BETA_FIXED + 1e-12
    return init, (low, high)


def self_bounds_and_init() -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    init = np.array(chess.Config.INITIAL_GUESS["selfconf"], dtype=float)
    low = chess.Config.BOUNDS_LOW_DEFAULT.copy()
    high = chess.Config.BOUNDS_HIGH_DEFAULT.copy()
    init[BETA_IDX] = chess.Config.BETA_FIXED
    low[BETA_IDX] = chess.Config.BETA_FIXED
    high[BETA_IDX] = chess.Config.BETA_FIXED + 1e-12
    return init, (low, high)


def fit_chess_slice(
    observed: np.ndarray,
    e_tensor: np.ndarray,
    align_tensor: np.ndarray,
    condition_array: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    use_self = model_name == "selfconf"
    init, bounds = self_bounds_and_init() if use_self else ai_bounds_and_init()
    fit = chess.fit_model(observed, e_tensor, align_tensor, init, bounds, use_self, condition_array)
    return {
        "gamma_res": float(fit["params"][GAMMA_IDX]),
        "params": fit["params"],
        "mse": float(fit["mse"]),
        "adj_r2": float(fit["adj_r2"]),
    }


def fit_chess_segment(
    observed_full: np.ndarray,
    e_full: np.ndarray,
    align_full: np.ndarray,
    condition_array: np.ndarray,
    start_trial: int,
    end_trial: int,
) -> dict[str, Any]:
    observed = observed_full[:, start_trial - 1 : end_trial + 1]
    e_tensor = e_full[:, start_trial - 1 : end_trial, :]
    align = align_full[:, start_trial - 1 : end_trial]
    return fit_chess_slice(observed, e_tensor, align, condition_array, "selfconf")


def valid_corr(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return {"n": int(mask.sum()), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    pr = pearsonr(x[mask], y[mask])
    sr = spearmanr(x[mask], y[mask])
    return {
        "n": int(mask.sum()),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def participant_fits(parts: list[chess.ParticipantData], matrices: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    condition_array = matrices["condition_array"]
    for i, part in enumerate(parts):
        one_cond = condition_array[i : i + 1]
        self_fit = fit_chess_slice(
            matrices["self_conf_matrix"][i : i + 1],
            matrices["e_tensor"][i : i + 1],
            matrices["align_tensor"][i : i + 1],
            one_cond,
            "selfconf",
        )
        ai_fit = fit_chess_slice(
            matrices["ai_conf_matrix"][i : i + 1],
            matrices["e_tensor"][i : i + 1],
            matrices["align_tensor"][i : i + 1],
            one_cond,
            "aiconf",
        )
        align_count = float(np.nansum(matrices["align_tensor"][i]))
        self_delta = np.diff(matrices["self_conf_matrix"][i])
        align = matrices["align_tensor"][i].astype(bool)
        aligned_abs_delta = float(np.nanmean(np.abs(self_delta[align]))) if np.any(align) else np.nan
        nonaligned_abs_delta = float(np.nanmean(np.abs(self_delta[~align]))) if np.any(~align) else np.nan
        rows.append(
            {
                "pid": int(part.pid),
                "condition": int(part.condition),
                "gamma_res_self": self_fit["gamma_res"],
                "gamma_res_ai": ai_fit["gamma_res"],
                "align_count": align_count,
                "aligned_self_abs_delta": aligned_abs_delta,
                "nonaligned_self_abs_delta": nonaligned_abs_delta,
                "aligned_minus_nonaligned_self_abs_delta": aligned_abs_delta - nonaligned_abs_delta,
                "self_mse": self_fit["mse"],
                "ai_mse": ai_fit["mse"],
            }
        )
    return pd.DataFrame(rows)


def simulate_self(
    params: np.ndarray,
    observed: np.ndarray,
    e_tensor: np.ndarray,
    align_tensor: np.ndarray,
    rng: np.random.Generator,
    noise_sd: float,
) -> np.ndarray:
    pred = chess.compute_predictions(params, observed, e_tensor, align_tensor, use_self_force=True)
    sim = pred.copy()
    sim[:, 1:] = np.clip(pred[:, 1:] + rng.normal(0.0, noise_sd, size=pred[:, 1:].shape), 0.0, 1.0)
    sim[:, 0] = observed[:, 0]
    return sim


def parameter_recovery(matrices: dict[str, np.ndarray], n_iter: int = 50) -> pd.DataFrame:
    init, bounds = self_bounds_and_init()
    fit = chess.fit_model(
        matrices["self_conf_matrix"],
        matrices["e_tensor"],
        matrices["align_tensor"],
        init,
        bounds,
        True,
        matrices["condition_array"],
    )
    pred = fit["pred"]
    noise_sd = float(np.nanstd(matrices["self_conf_matrix"][:, 1:] - pred[:, 1:], ddof=1))
    rng = np.random.default_rng(20260514)
    rows = []
    for iteration in range(1, n_iter + 1):
        sim_obs = simulate_self(fit["params"], matrices["self_conf_matrix"], matrices["e_tensor"], matrices["align_tensor"], rng, noise_sd)
        rec_fit = fit_chess_slice(sim_obs, matrices["e_tensor"], matrices["align_tensor"], matrices["condition_array"], "selfconf")
        rows.append(
            {
                "iteration": iteration,
                "true_gamma_res_self": float(fit["params"][GAMMA_IDX]),
                "recovered_gamma_res_self": rec_fit["gamma_res"],
                "recovered_adj_r2": rec_fit["adj_r2"],
                "noise_sd": noise_sd,
            }
        )
    return pd.DataFrame(rows)


def summarize_alignment_delta_chess(matrices: dict[str, np.ndarray]) -> pd.DataFrame:
    self_delta = np.diff(matrices["self_conf_matrix"], axis=1)
    ai_delta = np.diff(matrices["ai_conf_matrix"], axis=1)
    align = matrices["align_tensor"].astype(bool)
    rows = []
    for channel, delta in [("self", self_delta), ("AI", ai_delta)]:
        for status, mask in [("aligned", align), ("nonaligned", ~align)]:
            values = delta[mask]
            rows.append(
                {
                    "dataset": "chess",
                    "channel": channel,
                    "alignment_status": status,
                    "n": int(values.size),
                    "mean_signed_delta": float(np.nanmean(values)),
                    "mean_abs_delta": float(np.nanmean(np.abs(values))),
                }
            )
    return pd.DataFrame(rows)


def summarize_alignment_delta_truss() -> pd.DataFrame:
    matrices, _ = build_truss_matrices()
    align = matrices["align_matrix"].astype(bool)
    rows = []
    for channel, mat in [("self", matrices["self_conf_matrix"]), ("AI", matrices["ai_conf_matrix"])]:
        delta = np.diff(mat, axis=1)
        for status, mask in [("aligned", align), ("nonaligned", ~align)]:
            values = delta[mask]
            rows.append(
                {
                    "dataset": "truss",
                    "channel": channel,
                    "alignment_status": status,
                    "n": int(values.size),
                    "mean_signed_delta": float(np.nanmean(values)),
                    "mean_abs_delta": float(np.nanmean(np.abs(values))),
                }
            )
    return pd.DataFrame(rows)


def test_truss_condition_gamma() -> pd.DataFrame:
    matrices, _ = build_truss_matrices()
    rows = []
    for cond in sorted(np.unique(matrices["condition_array"])):
        idx = np.where(matrices["condition_array"] == cond)[0]
        sub = {
            k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == matrices["condition_array"].shape[:1] else v)
            for k, v in matrices.items()
        }
        e_tensor, align = tensors_for_variant(sub, "full_align")
        fit = fit_truss_full_model("selfconf", sub["self_conf_matrix"], e_tensor, align, sub["condition_array"], use_self_track=True)
        rows.append(
            {
                "dataset": "truss",
                "split": f"condition_{int(cond)}",
                "n_participants": int(idx.size),
                "gamma_res_self": float(fit["params"][7]),
                "adj_r2": float(fit["adj_r2"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts, _ = chess.load_participants(chess.Config.DATA_DIR)
    matrices = chess.build_matrices(parts)

    rows = []
    for cond in sorted(np.unique(matrices["condition_array"])):
        idx = np.where(matrices["condition_array"] == cond)[0]
        fit = fit_chess_slice(
            matrices["self_conf_matrix"][idx],
            matrices["e_tensor"][idx],
            matrices["align_tensor"][idx],
            matrices["condition_array"][idx],
            "selfconf",
        )
        rows.append({"dataset": "chess", "split": f"condition_{int(cond)}", "n_participants": int(idx.size), "gamma_res_self": fit["gamma_res"], "adj_r2": fit["adj_r2"]})

    for label, start, end in [("pre_trials_1_20", 1, 20), ("post_trials_21_30", 21, 30)]:
        fit = fit_chess_segment(matrices["self_conf_matrix"], matrices["e_tensor"], matrices["align_tensor"], matrices["condition_array"], start, end)
        rows.append({"dataset": "chess", "split": label, "n_participants": matrices["self_conf_matrix"].shape[0], "gamma_res_self": fit["gamma_res"], "adj_r2": fit["adj_r2"]})

    split_all = pd.concat([pd.DataFrame(rows), test_truss_condition_gamma()], ignore_index=True)
    split_all.to_csv(OUT_DIR / "gamma_res_self_split_stability.csv", index=False)

    per_participant = participant_fits(parts, matrices)
    per_participant.to_csv(OUT_DIR / "chess_individual_gamma_res_fits.csv", index=False)

    corr_rows = []
    for xcol, ycol, label in [
        ("gamma_res_self", "align_count", "gamma_res_self_vs_align_count"),
        ("gamma_res_self", "aligned_minus_nonaligned_self_abs_delta", "gamma_res_self_vs_alignment_abs_delta_contrast"),
        ("gamma_res_self", "gamma_res_ai", "gamma_res_self_vs_gamma_res_ai"),
    ]:
        corr_rows.append({"test": label, **valid_corr(per_participant[xcol].to_numpy(float), per_participant[ycol].to_numpy(float))})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_DIR / "chess_gamma_res_correlations.csv", index=False)

    recovery = parameter_recovery(matrices, n_iter=50)
    recovery.to_csv(OUT_DIR / "chess_gamma_res_parameter_recovery.csv", index=False)

    behavior = pd.concat([summarize_alignment_delta_chess(matrices), summarize_alignment_delta_truss()], ignore_index=True)
    behavior.to_csv(OUT_DIR / "alignment_confidence_delta_summary.csv", index=False)

    print("\nTEST 1/2: gamma_res_self split stability")
    print(split_all.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTEST 3/5: participant-level correlations")
    print(corr_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTEST 4: parameter recovery")
    print(
        recovery["recovered_gamma_res_self"]
        .agg(["mean", "std", "min", "max"])
        .to_frame("recovered_gamma_res_self")
        .to_string(float_format=lambda x: f"{x:.6f}")
    )
    print("\nTEST 6: alignment -> confidence change")
    print(behavior.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()

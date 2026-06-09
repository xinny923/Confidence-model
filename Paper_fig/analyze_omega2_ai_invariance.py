from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "Paper_fig"
TRUSS_DIR = ROOT / "truss_data"
OUT_DIR = PAPER_DIR / "full_output" / "omega2_invariance"

if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))
if str(TRUSS_DIR) not in sys.path:
    sys.path.insert(0, str(TRUSS_DIR))

import full_model as chess  # noqa: E402
from truss_detailed_subset_fit import build_subset_matrices as build_truss_matrices  # noqa: E402
from truss_detailed_subset_fit import fit_full_model as fit_truss_full_model  # noqa: E402
from truss_detailed_subset_fit import tensors_for_variant as truss_tensors_for_variant  # noqa: E402
from analyze_alignment_reliance_distribution import positive_feedback_column  # noqa: E402


PARAMS = chess.Config.PARAMETER_NAMES
OMEGA2_IDX = PARAMS.index("omega2")
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
    model_name: str = "aiconf",
) -> dict[str, Any]:
    use_self = model_name == "selfconf"
    init, bounds = self_bounds_and_init() if use_self else ai_bounds_and_init()
    fit = chess.fit_model(observed, e_tensor, align_tensor, init, bounds, use_self, condition_array)
    return {
        "omega2": float(fit["params"][OMEGA2_IDX]),
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
    # Trials are 1-indexed; observed includes trial 0 baseline at column 0.
    observed = observed_full[:, start_trial - 1 : end_trial + 1]
    e_tensor = e_full[:, start_trial - 1 : end_trial, :]
    align = align_full[:, start_trial - 1 : end_trial]
    return fit_chess_slice(observed, e_tensor, align, condition_array, "aiconf")


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
        ai_fit = fit_chess_slice(
            matrices["ai_conf_matrix"][i : i + 1],
            matrices["e_tensor"][i : i + 1],
            matrices["align_tensor"][i : i + 1],
            one_cond,
            "aiconf",
        )
        self_fit = fit_chess_slice(
            matrices["self_conf_matrix"][i : i + 1],
            matrices["e_tensor"][i : i + 1],
            matrices["align_tensor"][i : i + 1],
            one_cond,
            "selfconf",
        )
        reject_positive = sum(
            1
            for rec in part.trial_records
            if rec["action_label"] == "reject" and float(rec["feedback"]) > 0
        )
        rows.append(
            {
                "pid": int(part.pid),
                "condition": int(part.condition),
                "omega2_ai": ai_fit["omega2"],
                "omega2_self": self_fit["omega2"],
                "reject_positive_count": int(reject_positive),
                "ai_mse": ai_fit["mse"],
                "self_mse": self_fit["mse"],
            }
        )
    return pd.DataFrame(rows)


def simulate_chess_ai(
    params: np.ndarray,
    observed: np.ndarray,
    e_tensor: np.ndarray,
    align_tensor: np.ndarray,
    rng: np.random.Generator,
    noise_sd: float,
) -> np.ndarray:
    pred = chess.compute_predictions(params, observed, e_tensor, align_tensor, use_self_force=False)
    sim = pred.copy()
    noise = rng.normal(0.0, noise_sd, size=pred[:, 1:].shape)
    sim[:, 1:] = np.clip(pred[:, 1:] + noise, 0.0, 1.0)
    sim[:, 0] = observed[:, 0]
    return sim


def parameter_recovery(matrices: dict[str, np.ndarray], n_iter: int = 50) -> pd.DataFrame:
    init, bounds = ai_bounds_and_init()
    fit = chess.fit_model(
        matrices["ai_conf_matrix"],
        matrices["e_tensor"],
        matrices["align_tensor"],
        init,
        bounds,
        False,
        matrices["condition_array"],
    )
    pred = fit["pred"]
    residual = matrices["ai_conf_matrix"][:, 1:] - pred[:, 1:]
    noise_sd = float(np.nanstd(residual, ddof=1))

    rng = np.random.default_rng(20260513)
    rows = []
    for iteration in range(1, n_iter + 1):
        sim_obs = simulate_chess_ai(
            fit["params"],
            matrices["ai_conf_matrix"],
            matrices["e_tensor"],
            matrices["align_tensor"],
            rng,
            noise_sd,
        )
        rec_fit = fit_chess_slice(sim_obs, matrices["e_tensor"], matrices["align_tensor"], matrices["condition_array"], "aiconf")
        rows.append(
            {
                "iteration": iteration,
                "true_omega2_ai": float(fit["params"][OMEGA2_IDX]),
                "recovered_omega2_ai": rec_fit["omega2"],
                "recovered_adj_r2": rec_fit["adj_r2"],
                "noise_sd": noise_sd,
            }
        )
    return pd.DataFrame(rows)


def next_trial_accept_chess(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in trials.sort_values(["pid", "trial"]).groupby("pid"):
        records = group.to_dict("records")
        for rec, nxt in zip(records[:-1], records[1:]):
            is_reject_positive = rec["action_label"] == "reject" and float(rec["feedback"]) > 0
            rows.append(
                {
                    "dataset": "chess",
                    "trigger": "reject_positive" if is_reject_positive else "other",
                    "next_accept": 1.0 if nxt["action_label"] == "accept" else 0.0,
                }
            )
    return pd.DataFrame(rows)


def next_trial_accept_truss() -> pd.DataFrame:
    recon = pd.read_csv(TRUSS_DIR / "data_folder_output" / "data_folder_reconstructed_trials.csv")
    usable = recon[(recon["has_move1"]) & (recon["has_move2"]) & (recon["has_suggestion"]) & (recon["reconstructed_action"] != "missing")].copy()
    usable["positive_feedback"] = positive_feedback_column(usable)
    rows = []
    for _, group in usable.sort_values(["participant", "trial"]).groupby("participant"):
        records = group.to_dict("records")
        for rec, nxt in zip(records[:-1], records[1:]):
            is_reject_positive = rec["reconstructed_action"] == "reject_ai" and bool(rec["positive_feedback"])
            rows.append(
                {
                    "dataset": "truss",
                    "trigger": "reject_positive" if is_reject_positive else "other",
                    "next_accept": 1.0 if nxt["reconstructed_action"] == "accept_ai" else 0.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_next_accept(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, dset in df.groupby("dataset"):
        rates = {}
        for trigger, group in dset.groupby("trigger"):
            rates[trigger] = float(group["next_accept"].mean())
            rows.append({"dataset": dataset, "trigger": trigger, "n": int(group.shape[0]), "next_accept_rate": rates[trigger]})
        if "reject_positive" in rates and "other" in rates:
            rows.append(
                {
                    "dataset": dataset,
                    "trigger": "reject_positive_minus_other",
                    "n": np.nan,
                    "next_accept_rate": rates["reject_positive"] - rates["other"],
                }
            )
    return pd.DataFrame(rows)


def test_truss_condition_omega2() -> pd.DataFrame:
    matrices, _ = build_truss_matrices()
    rows = []
    for cond in sorted(np.unique(matrices["condition_array"])):
        idx = np.where(matrices["condition_array"] == cond)[0]
        sub = {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == matrices["condition_array"].shape[:1] else v) for k, v in matrices.items()}
        e_tensor, align = truss_tensors_for_variant(sub, "full_align")
        fit = fit_truss_full_model("aiconf", sub["ai_conf_matrix"], e_tensor, align, sub["condition_array"], use_self_track=False)
        rows.append(
            {
                "dataset": "truss",
                "split": f"condition_{int(cond)}",
                "n_participants": int(idx.size),
                "omega2_ai": float(fit["params"][4]),
                "adj_r2": float(fit["adj_r2"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts, trials = chess.load_participants(chess.Config.DATA_DIR)
    matrices = chess.build_matrices(parts)

    rows = []
    for cond in sorted(np.unique(matrices["condition_array"])):
        idx = np.where(matrices["condition_array"] == cond)[0]
        fit = fit_chess_slice(
            matrices["ai_conf_matrix"][idx],
            matrices["e_tensor"][idx],
            matrices["align_tensor"][idx],
            matrices["condition_array"][idx],
            "aiconf",
        )
        rows.append({"dataset": "chess", "split": f"condition_{int(cond)}", "n_participants": int(idx.size), "omega2_ai": fit["omega2"], "adj_r2": fit["adj_r2"]})

    for label, start, end in [("pre_trials_1_20", 1, 20), ("post_trials_21_30", 21, 30)]:
        fit = fit_chess_segment(
            matrices["ai_conf_matrix"],
            matrices["e_tensor"],
            matrices["align_tensor"],
            matrices["condition_array"],
            start,
            end,
        )
        rows.append({"dataset": "chess", "split": label, "n_participants": matrices["ai_conf_matrix"].shape[0], "omega2_ai": fit["omega2"], "adj_r2": fit["adj_r2"]})

    split_df = pd.DataFrame(rows)
    truss_cond_df = test_truss_condition_omega2()
    split_all = pd.concat([split_df, truss_cond_df], ignore_index=True)
    split_all.to_csv(OUT_DIR / "omega2_ai_split_stability.csv", index=False)

    per_participant = participant_fits(parts, matrices)
    per_participant.to_csv(OUT_DIR / "chess_individual_omega2_fits.csv", index=False)

    corr_rows = []
    for xcol, ycol, label in [
        ("omega2_ai", "reject_positive_count", "omega2_ai_vs_reject_positive_count"),
        ("omega2_ai", "omega2_self", "omega2_ai_vs_omega2_self"),
    ]:
        result = valid_corr(per_participant[xcol].to_numpy(float), per_participant[ycol].to_numpy(float))
        corr_rows.append({"test": label, **result})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_DIR / "chess_omega2_correlations.csv", index=False)

    recovery = parameter_recovery(matrices, n_iter=50)
    recovery.to_csv(OUT_DIR / "chess_omega2_parameter_recovery.csv", index=False)

    behavior = pd.concat([next_trial_accept_chess(trials), next_trial_accept_truss()], ignore_index=True)
    behavior_summary = summarize_next_accept(behavior)
    behavior.to_csv(OUT_DIR / "reject_positive_next_trial_accept_raw.csv", index=False)
    behavior_summary.to_csv(OUT_DIR / "reject_positive_next_trial_accept_summary.csv", index=False)

    print("\nTEST 1/2: omega2_AI split stability")
    print(split_all.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTEST 3/5: participant-level correlations")
    print(corr_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTEST 4: parameter recovery")
    print(
        recovery["recovered_omega2_ai"]
        .agg(["mean", "std", "min", "max"])
        .to_frame("recovered_omega2_ai")
        .to_string(float_format=lambda x: f"{x:.6f}")
    )
    print("\nTEST 6: reject-positive -> next-trial accept")
    print(behavior_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()

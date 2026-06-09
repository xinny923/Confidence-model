from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "Paper_fig"
OUT_DIR = PAPER_DIR / "manu_fig"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def params_from_csv(path: Path, track: str, names: list[str]) -> np.ndarray:
    df = pd.read_csv(path)
    row = df[df["model"] == track].iloc[0]
    return row[names].to_numpy(dtype=float)


def mean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def channel_trial_stats(
    observed: np.ndarray,
    pred_base: np.ndarray,
    pred_full: np.ndarray,
    align: np.ndarray,
    channel: str,
) -> pd.DataFrame:
    rows = []
    for t in range(1, observed.shape[1]):
        mask = align[:, t - 1].astype(bool)
        obs_t = observed[mask, t]
        base_t = pred_base[mask, t]
        full_t = pred_full[mask, t]

        base_abs_err = np.abs(obs_t - base_t)
        full_abs_err = np.abs(obs_t - full_t)
        improvement = base_abs_err - full_abs_err

        obs_delta = np.abs(observed[mask, t] - observed[mask, t - 1])
        base_delta = np.abs(pred_base[mask, t] - pred_base[mask, t - 1])
        full_delta = np.abs(pred_full[mask, t] - pred_full[mask, t - 1])

        rows.append({
            "channel": channel,
            "trial": t,
            "align_count": int(mask.sum()),
            "baseline_abs_error": mean_or_nan(base_abs_err),
            "full_abs_error": mean_or_nan(full_abs_err),
            "mean_improvement": mean_or_nan(improvement),
            "empirical_abs_delta": mean_or_nan(obs_delta),
            "baseline_abs_delta": mean_or_nan(base_delta),
            "full_abs_delta": mean_or_nan(full_delta),
        })
    return pd.DataFrame(rows)


def channel_case_stats(
    observed: np.ndarray,
    pred_base: np.ndarray,
    pred_full: np.ndarray,
    align: np.ndarray,
    channel: str,
) -> pd.DataFrame:
    rows = []
    align_counts = align.sum(axis=0).astype(int)
    for t in range(1, observed.shape[1]):
        mask = align[:, t - 1].astype(bool)
        for pid in np.where(mask)[0]:
            base_abs_err = abs(observed[pid, t] - pred_base[pid, t])
            full_abs_err = abs(observed[pid, t] - pred_full[pid, t])
            rows.append({
                "channel": channel,
                "participant_index": int(pid),
                "trial": int(t),
                "align_count": int(align_counts[t - 1]),
                "improvement": float(base_abs_err - full_abs_err),
                "empirical_abs_delta": float(abs(observed[pid, t] - observed[pid, t - 1])),
            })
    return pd.DataFrame(rows)


def correlation_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel, sub in df.groupby("channel", sort=False):
        clean = sub.dropna(subset=["mean_improvement", "align_count", "empirical_abs_delta"])
        for x_col in ["align_count", "empirical_abs_delta"]:
            x = clean[x_col].to_numpy(dtype=float)
            y = clean["mean_improvement"].to_numpy(dtype=float)
            pearson = stats.pearsonr(x, y)
            spearman = stats.spearmanr(x, y)
            rows.append({
                "channel": channel,
                "x": x_col,
                "y": "mean_improvement",
                "n_trials": int(len(clean)),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            })
    return pd.DataFrame(rows)


def ols_interaction_rows(case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel, sub in case_df.groupby("channel", sort=False):
        clean = sub.dropna(subset=["improvement", "align_count", "empirical_abs_delta"]).copy()
        clean["interaction"] = clean["empirical_abs_delta"] * clean["align_count"]
        y = clean["improvement"].to_numpy(dtype=float)
        x = clean[["empirical_abs_delta", "align_count", "interaction"]].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(y)), x])

        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        resid = y - x @ beta
        n = len(y)
        p = x.shape[1]
        sigma2 = float(resid @ resid) / (n - p)
        cov = sigma2 * np.linalg.inv(x.T @ x)
        se = np.sqrt(np.diag(cov))
        t_stat = beta / se
        p_val = 2 * stats.t.sf(np.abs(t_stat), df=n - p)
        for name, b, s, t, pv in zip(["intercept", "empirical_abs_delta", "align_count", "interaction"], beta, se, t_stat, p_val):
            rows.append({
                "channel": channel,
                "term": name,
                "n_alignment_cases": int(n),
                "coef": float(b),
                "se": float(s),
                "t": float(t),
                "p": float(pv),
            })
    return pd.DataFrame(rows)


def main() -> None:
    og = load_module("sec5_og_physics", PAPER_DIR / "og_physics.py")
    full = load_module("sec5_full_model", PAPER_DIR / "full_model.py")

    base_parts, _ = og.load_participants(og.Config.DATA_DIR)
    base_mats = og.build_matrices(base_parts)
    full_parts, _ = full.load_participants(full.Config.DATA_DIR)
    full_mats = full.build_matrices(full_parts)

    base_ai_p = params_from_csv(PAPER_DIR / "og_physics_output" / "og_physics_params.csv", "aiconf", og.Config.PARAMETER_NAMES)
    base_self_p = params_from_csv(PAPER_DIR / "og_physics_output" / "og_physics_params.csv", "selfconf", og.Config.PARAMETER_NAMES)
    full_ai_p = params_from_csv(PAPER_DIR / "full_output" / "full_model_params.csv", "aiconf", full.Config.PARAMETER_NAMES)
    full_self_p = params_from_csv(PAPER_DIR / "full_output" / "full_model_params.csv", "selfconf", full.Config.PARAMETER_NAMES)

    pred_base_ai = og.compute_predictions(base_ai_p, base_mats["ai_conf_matrix"], base_mats["e_tensor"])
    pred_base_self = og.compute_predictions(base_self_p, base_mats["self_conf_matrix"], base_mats["e_tensor"])
    pred_full_ai = full.compute_predictions(
        full_ai_p,
        full_mats["ai_conf_matrix"],
        full_mats["e_tensor"],
        full_mats["align_tensor"],
        False,
    )
    pred_full_self = full.compute_predictions(
        full_self_p,
        full_mats["self_conf_matrix"],
        full_mats["e_tensor"],
        full_mats["align_tensor"],
        True,
    )

    trial_df = pd.concat([
        channel_trial_stats(
            full_mats["ai_conf_matrix"],
            pred_base_ai,
            pred_full_ai,
            full_mats["align_tensor"],
            "AI",
        ),
        channel_trial_stats(
            full_mats["self_conf_matrix"],
            pred_base_self,
            pred_full_self,
            full_mats["align_tensor"],
            "Self",
        ),
    ], ignore_index=True)
    case_df = pd.concat([
        channel_case_stats(
            full_mats["ai_conf_matrix"],
            pred_base_ai,
            pred_full_ai,
            full_mats["align_tensor"],
            "AI",
        ),
        channel_case_stats(
            full_mats["self_conf_matrix"],
            pred_base_self,
            pred_full_self,
            full_mats["align_tensor"],
            "Self",
        ),
    ], ignore_index=True)
    corr_df = correlation_rows(trial_df)
    ols_df = ols_interaction_rows(case_df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trial_path = OUT_DIR / "sec5_alignment_trial_stats.csv"
    case_path = OUT_DIR / "sec5_alignment_case_stats.csv"
    corr_path = OUT_DIR / "sec5_alignment_correlations.csv"
    ols_path = OUT_DIR / "sec5_alignment_interaction_regression.csv"
    trial_df.to_csv(trial_path, index=False)
    case_df.to_csv(case_path, index=False)
    corr_df.to_csv(corr_path, index=False)
    ols_df.to_csv(ols_path, index=False)

    print(f"Saved {trial_path}")
    print(f"Saved {case_path}")
    print(f"Saved {corr_path}")
    print(f"Saved {ols_path}")
    print(corr_df.to_string(index=False))
    print()
    print(ols_df.to_string(index=False))


if __name__ == "__main__":
    main()

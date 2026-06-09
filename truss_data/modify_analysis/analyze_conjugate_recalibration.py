"""Tests for non-monotonic confidence recalibration in truss data.

The focus is on split confidence updates:

- AI_up_self_down: AI-confidence increases while self-confidence decreases.
- AI_down_self_up: AI-confidence decreases while self-confidence increases.

The analyses test whether split updates are associated with prior imbalance
between the two confidence channels, and whether modify trials show stronger
AI/self confidence coupling than non-modify trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parent
TRIAL_PATH = ROOT / "trial_level_modify_analysis.csv"
RNG_SEED = 42
N_BOOT = 300

MONOTONIC = {"AI_up_self_up", "AI_down_self_down"}
SPLIT = {"AI_up_self_down", "AI_down_self_up"}
NONZERO = MONOTONIC | SPLIT


@dataclass
class LogisticSpec:
    name: str
    outcome: str
    predictors: List[str]


def prepare_data() -> pd.DataFrame:
    df = pd.read_csv(TRIAL_PATH)
    df["prior_gap_self_minus_ai"] = df["self_before"] - df["ai_before"]
    df["abs_prior_gap"] = df["prior_gap_self_minus_ai"].abs()
    df["condition2"] = (df["condition"] == 2).astype(float)
    df["trial_z"] = (df["trial"] - df["trial"].mean()) / df["trial"].std(ddof=0)
    df["action_reject"] = (df["action"] == "reject_ai").astype(float)
    df["action_modify"] = (df["action"] == "modify").astype(float)
    df["negative_feedback"] = 1.0 - df["positive_feedback"].astype(float)
    df["reject_x_negative"] = df["action_reject"] * df["negative_feedback"]
    df["is_nonzero"] = df["quadrant"].isin(NONZERO)
    df["is_split"] = df["quadrant"].isin(SPLIT).astype(float)
    df["is_ai_up_self_down"] = (df["quadrant"] == "AI_up_self_down").astype(float)
    return df


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def build_design(data: pd.DataFrame, predictors: List[str]) -> Tuple[np.ndarray, List[str]]:
    x_main = data[predictors].to_numpy(dtype=float)
    participants = data["participant"].astype(int).to_numpy()
    unique_pids = sorted(np.unique(participants))
    # Drop the first participant dummy to keep the intercept identifiable.
    dummy_cols = []
    dummy_names = []
    for pid in unique_pids[1:]:
        dummy_cols.append((participants == pid).astype(float))
        dummy_names.append(f"participant_{pid}")
    if dummy_cols:
        x_fe = np.column_stack(dummy_cols)
        x = np.column_stack([np.ones(data.shape[0]), x_main, x_fe])
    else:
        x = np.column_stack([np.ones(data.shape[0]), x_main])
    names = ["intercept", *predictors, *dummy_names]
    return x, names


def fit_logistic_fixed_effects(
    data: pd.DataFrame,
    outcome: str,
    predictors: List[str],
    participant_penalty: float = 0.01,
) -> Tuple[pd.Series, float]:
    work = data[[outcome, "participant", *predictors]].dropna().reset_index(drop=True)
    y = work[outcome].to_numpy(dtype=float)
    x, names = build_design(work, predictors)
    n_main = 1 + len(predictors)
    penalty_mask = np.zeros(x.shape[1], dtype=float)
    penalty_mask[n_main:] = participant_penalty

    def objective(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        eta = x @ beta
        p = sigmoid(eta)
        eps = 1e-12
        nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        penalty = 0.5 * np.sum(penalty_mask * beta * beta)
        grad = x.T @ (p - y) + penalty_mask * beta
        return float(nll + penalty), grad

    init = np.zeros(x.shape[1], dtype=float)
    result = optimize.minimize(
        fun=lambda b: objective(b)[0],
        x0=init,
        jac=lambda b: objective(b)[1],
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        # Retry with a slightly stronger participant penalty if separation bites.
        return fit_logistic_fixed_effects(data, outcome, predictors, participant_penalty=0.1)
    return pd.Series(result.x, index=names), float(result.fun)


def bootstrap_logistic(
    data: pd.DataFrame,
    spec: LogisticSpec,
    n_boot: int = N_BOOT,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    participants = np.array(sorted(data["participant"].unique()))
    rows = []
    for iteration in range(1, n_boot + 1):
        sampled = rng.choice(participants, size=participants.shape[0], replace=True)
        pieces = []
        for draw_idx, pid in enumerate(sampled, start=1):
            part = data[data["participant"] == pid].copy()
            # Make bootstrap participant clusters unique for fixed effects.
            part["participant"] = draw_idx
            pieces.append(part)
        boot = pd.concat(pieces, ignore_index=True)
        try:
            beta, _ = fit_logistic_fixed_effects(boot, spec.outcome, spec.predictors)
        except RecursionError:
            continue
        row = {"model": spec.name, "iteration": iteration}
        row.update({name: float(beta.get(name, np.nan)) for name in spec.predictors})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_bootstrap(
    observed: pd.Series,
    boot: pd.DataFrame,
    spec: LogisticSpec,
) -> pd.DataFrame:
    rows = []
    for predictor in spec.predictors:
        values = boot[predictor].dropna().to_numpy(dtype=float)
        obs = float(observed[predictor])
        if values.size:
            ci_low, ci_high = np.percentile(values, [2.5, 97.5])
            one_side = min(np.mean(values <= 0), np.mean(values >= 0))
            boot_p = min(1.0, 2.0 * ((one_side * values.size + 1.0) / (values.size + 1.0)))
        else:
            ci_low = ci_high = boot_p = np.nan
        rows.append({
            "model": spec.name,
            "predictor": predictor,
            "coef": obs,
            "odds_ratio": float(np.exp(obs)),
            "boot_ci_low": float(ci_low),
            "boot_ci_high": float(ci_high),
            "bootstrap_p": float(boot_p),
            "n_boot": int(values.size),
        })
    return pd.DataFrame(rows)


def run_logistic_models(df: pd.DataFrame) -> None:
    nonzero = df[df["is_nonzero"]].copy()
    split_only = nonzero[nonzero["quadrant"].isin(SPLIT)].copy()

    specs = [
        LogisticSpec(
            name="split_vs_monotonic",
            outcome="is_split",
            predictors=[
                "prior_gap_self_minus_ai",
                "positive_feedback",
                "action_reject",
                "action_modify",
                "condition2",
                "trial_z",
            ],
        ),
        LogisticSpec(
            name="ai_up_self_down_vs_ai_down_self_up",
            outcome="is_ai_up_self_down",
            predictors=[
                "prior_gap_self_minus_ai",
                "positive_feedback",
                "action_reject",
                "action_modify",
                "condition2",
                "trial_z",
            ],
        ),
        LogisticSpec(
            name="ai_up_self_down_vs_other_nonzero",
            outcome="is_ai_up_self_down",
            predictors=[
                "prior_gap_self_minus_ai",
                "action_reject",
                "negative_feedback",
                "reject_x_negative",
                "condition2",
                "trial_z",
            ],
        ),
    ]
    data_by_model = {
        "split_vs_monotonic": nonzero,
        "ai_up_self_down_vs_ai_down_self_up": split_only,
        "ai_up_self_down_vs_other_nonzero": nonzero,
    }

    summary_tables = []
    boot_tables = []
    for offset, spec in enumerate(specs):
        data = data_by_model[spec.name]
        observed, objective = fit_logistic_fixed_effects(data, spec.outcome, spec.predictors)
        boot = bootstrap_logistic(data, spec, seed=RNG_SEED + offset)
        boot_tables.append(boot)
        summary = summarize_bootstrap(observed, boot, spec)
        summary.insert(1, "n_trials", int(data.shape[0]))
        summary.insert(2, "n_participants", int(data["participant"].nunique()))
        summary.insert(3, "objective", objective)
        summary_tables.append(summary)

    pd.concat(summary_tables, ignore_index=True).to_csv(
        ROOT / "conjugate_logistic_bootstrap_summary.csv",
        index=False,
    )
    pd.concat(boot_tables, ignore_index=True).to_csv(
        ROOT / "conjugate_logistic_bootstrap_raw.csv",
        index=False,
    )


def bootstrap_mean_difference(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    positive_group: str,
    negative_group: str,
    n_boot: int = N_BOOT,
    seed: int = RNG_SEED,
) -> Tuple[float, float, float, float]:
    observed = (
        data[data[group_col] == positive_group][value_col].mean()
        - data[data[group_col] == negative_group][value_col].mean()
    )
    rng = np.random.default_rng(seed)
    participants = np.array(sorted(data["participant"].unique()))
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(participants, size=participants.shape[0], replace=True)
        boot = pd.concat([data[data["participant"] == pid] for pid in sampled], ignore_index=True)
        diff = (
            boot[boot[group_col] == positive_group][value_col].mean()
            - boot[boot[group_col] == negative_group][value_col].mean()
        )
        diffs.append(float(diff))
    arr = np.asarray(diffs, dtype=float)
    ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
    one_side = min(np.mean(arr <= 0), np.mean(arr >= 0))
    boot_p = min(1.0, 2.0 * ((one_side * arr.size + 1.0) / (arr.size + 1.0)))
    return float(observed), float(ci_low), float(ci_high), float(boot_p)


def run_abs_imbalance_test(df: pd.DataFrame) -> None:
    nonzero = df[df["is_nonzero"]].copy()
    nonzero["update_type"] = np.where(nonzero["is_split"] == 1, "split", "monotonic")
    split = nonzero[nonzero["update_type"] == "split"]["abs_prior_gap"]
    monotonic = nonzero[nonzero["update_type"] == "monotonic"]["abs_prior_gap"]
    welch = stats.ttest_ind(split, monotonic, equal_var=False)
    diff, ci_low, ci_high, boot_p = bootstrap_mean_difference(
        nonzero,
        "abs_prior_gap",
        "update_type",
        "split",
        "monotonic",
    )
    pd.DataFrame([{
        "comparison": "abs_prior_gap_split_minus_monotonic",
        "split_n": int(split.shape[0]),
        "monotonic_n": int(monotonic.shape[0]),
        "split_mean": float(split.mean()),
        "monotonic_mean": float(monotonic.mean()),
        "difference": diff,
        "welch_t": float(welch.statistic),
        "welch_p": float(welch.pvalue),
        "cluster_boot_ci_low": ci_low,
        "cluster_boot_ci_high": ci_high,
        "cluster_boot_p": boot_p,
        "n_boot": N_BOOT,
    }]).to_csv(ROOT / "split_vs_monotonic_abs_gap_test.csv", index=False)


def pearson_safe(data: pd.DataFrame) -> float:
    valid = data[["delta_ai", "delta_self"]].dropna()
    if valid.shape[0] < 3:
        return np.nan
    if valid["delta_ai"].std(ddof=0) == 0 or valid["delta_self"].std(ddof=0) == 0:
        return np.nan
    return float(stats.pearsonr(valid["delta_ai"], valid["delta_self"]).statistic)


def fisher_z(r: float) -> float:
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))


def run_modify_coupling_bootstrap(df: pd.DataFrame) -> None:
    modify = df[df["action"] == "modify"]
    non_modify = df[df["action"] != "modify"]
    observed_modify_r = pearson_safe(modify)
    observed_non_modify_r = pearson_safe(non_modify)
    observed_diff_r = observed_modify_r - observed_non_modify_r
    observed_diff_z = fisher_z(observed_modify_r) - fisher_z(observed_non_modify_r)

    rng = np.random.default_rng(RNG_SEED)
    participants = np.array(sorted(df["participant"].unique()))
    rows = []
    for iteration in range(1, N_BOOT + 1):
        sampled = rng.choice(participants, size=participants.shape[0], replace=True)
        boot = pd.concat([df[df["participant"] == pid] for pid in sampled], ignore_index=True)
        r_mod = pearson_safe(boot[boot["action"] == "modify"])
        r_non = pearson_safe(boot[boot["action"] != "modify"])
        if np.isnan(r_mod) or np.isnan(r_non):
            continue
        rows.append({
            "iteration": iteration,
            "modify_r": r_mod,
            "non_modify_r": r_non,
            "diff_r": r_mod - r_non,
            "diff_fisher_z": fisher_z(r_mod) - fisher_z(r_non),
        })
    raw = pd.DataFrame(rows)
    raw.to_csv(ROOT / "modify_vs_nonmodify_coupling_bootstrap_raw.csv", index=False)

    diff = raw["diff_r"].to_numpy(dtype=float)
    diff_z = raw["diff_fisher_z"].to_numpy(dtype=float)
    ci_low, ci_high = np.percentile(diff, [2.5, 97.5])
    z_low, z_high = np.percentile(diff_z, [2.5, 97.5])
    one_side = min(np.mean(diff <= 0), np.mean(diff >= 0))
    boot_p = min(1.0, 2.0 * ((one_side * diff.size + 1.0) / (diff.size + 1.0)))
    pd.DataFrame([{
        "comparison": "modify_minus_non_modify_ai_self_delta_correlation",
        "modify_r": observed_modify_r,
        "non_modify_r": observed_non_modify_r,
        "diff_r": observed_diff_r,
        "diff_r_ci_low": float(ci_low),
        "diff_r_ci_high": float(ci_high),
        "diff_fisher_z": observed_diff_z,
        "diff_fisher_z_ci_low": float(z_low),
        "diff_fisher_z_ci_high": float(z_high),
        "bootstrap_p": float(boot_p),
        "n_boot": int(diff.size),
    }]).to_csv(ROOT / "modify_vs_nonmodify_coupling_bootstrap_summary.csv", index=False)


def main() -> None:
    df = prepare_data()
    run_logistic_models(df)
    run_abs_imbalance_test(df)
    run_modify_coupling_bootstrap(df)
    print(f"Saved conjugate recalibration analyses to {ROOT}")


if __name__ == "__main__":
    main()

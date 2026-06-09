from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
CHESS_SELF_PATH = ROOT / "Paper_fig" / "full_output" / "robust_params_selfconf.csv"
CHESS_AI_PATH = ROOT / "Paper_fig" / "full_output" / "robust_params_aiconf.csv"
TRUSS_PATH = ROOT / "truss_data" / "data_folder_output" / "detailed_available_robust_raw.csv"
OUT_DIR = ROOT / "truss_data" / "data_folder_output"

PARAMS = [
    "alpha_fast",
    "alpha_slow",
    "alpha_a",
    "omega1",
    "omega2",
    "omega3",
    "omega4",
    "gamma_res",
]
EQUIVALENCE_DELTA = 0.10


def bootstrap_ci_diff(chess: np.ndarray, truss: np.ndarray, n_boot: int = 20000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        c = rng.choice(chess, size=chess.size, replace=True)
        t = rng.choice(truss, size=truss.size, replace=True)
        diffs[i] = t.mean() - c.mean()
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def permutation_pvalue_1d(chess: np.ndarray, truss: np.ndarray, n_perm: int = 20000, seed: int = 17) -> float:
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([chess, truss])
    n_chess = chess.size
    observed = abs(truss.mean() - chess.mean())
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        stat = abs(perm[n_chess:].mean() - perm[:n_chess].mean())
        count += stat >= observed
    return (count + 1) / (n_perm + 1)


def permutation_pvalue_vector(chess: np.ndarray, truss: np.ndarray, n_perm: int = 50000, seed: int = 23) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pooled = np.vstack([chess, truss])
    scale = pooled.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    n_chess = chess.shape[0]

    def stat(a: np.ndarray, b: np.ndarray) -> float:
        diff = (b.mean(axis=0) - a.mean(axis=0)) / scale
        return float(np.linalg.norm(diff))

    observed = stat(chess, truss)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(pooled.shape[0])
        a = pooled[idx[:n_chess]]
        b = pooled[idx[n_chess:]]
        count += stat(a, b) >= observed
    return observed, (count + 1) / (n_perm + 1)


def welch_df(var_a: float, n_a: int, var_b: float, n_b: int) -> float:
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = (var_a**2 / (n_a**2 * (n_a - 1))) + (var_b**2 / (n_b**2 * (n_b - 1)))
    return float(numerator / denominator)


def tost_equivalence(chess: np.ndarray, truss: np.ndarray, delta: float = EQUIVALENCE_DELTA) -> dict[str, float | bool]:
    """Welch two one-sided tests for equivalence of mean difference."""
    n_c, n_t = chess.size, truss.size
    mean_diff = truss.mean() - chess.mean()
    var_c = chess.var(ddof=1)
    var_t = truss.var(ddof=1)
    se = np.sqrt(var_c / n_c + var_t / n_t)
    df = welch_df(var_c, n_c, var_t, n_t)

    # H01: diff <= -delta vs diff > -delta
    t_lower = (mean_diff + delta) / se
    p_lower = 1.0 - stats.t.cdf(t_lower, df)
    # H02: diff >= +delta vs diff < +delta
    t_upper = (mean_diff - delta) / se
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)
    return {
        "tost_delta": delta,
        "tost_t_lower": float(t_lower),
        "tost_p_lower": float(p_lower),
        "tost_t_upper": float(t_upper),
        "tost_p_upper": float(p_upper),
        "tost_p": float(p_tost),
        "tost_equivalent": bool(p_tost < 0.05),
    }


def bic_bayes_factor_equal_vs_independent(chess: np.ndarray, truss: np.ndarray) -> dict[str, float]:
    """BIC approximation to BF for separate means vs shared mean.

    M0/equality: one shared mean, group-specific variances.
    M1/difference: separate group means, group-specific variances.
    BF10 supports M1 (different means); BF01 supports M0 (shared mean).
    """
    n_c, n_t = chess.size, truss.size
    n = n_c + n_t

    def normal_loglik(values: np.ndarray, mean: float) -> float:
        residual = values - mean
        sigma2 = float(np.mean(residual**2))
        sigma2 = max(sigma2, 1e-15)
        return float(-0.5 * values.size * (np.log(2 * np.pi * sigma2) + 1.0))

    shared_mean = float((chess.sum() + truss.sum()) / n)
    ll_equal = normal_loglik(chess, shared_mean) + normal_loglik(truss, shared_mean)
    ll_ind = normal_loglik(chess, float(chess.mean())) + normal_loglik(truss, float(truss.mean()))

    # group-specific variances in both models. Parameters:
    # equal: shared mean + two variances = 3; independent: two means + two variances = 4.
    bic_equal = 3 * np.log(n) - 2 * ll_equal
    bic_ind = 4 * np.log(n) - 2 * ll_ind
    bf10 = float(np.exp((bic_equal - bic_ind) / 2.0))
    return {
        "bic_equal": float(bic_equal),
        "bic_independent": float(bic_ind),
        "bf10_independent_over_equal": bf10,
        "bf01_equal_over_independent": float(1.0 / bf10) if bf10 > 0 else float("inf"),
    }


def compare_channel(channel: str, chess_path: Path, truss_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    chess = pd.read_csv(chess_path)[PARAMS].astype(float)
    truss_raw = pd.read_csv(TRUSS_PATH)
    truss = truss_raw[truss_raw["Model"] == "PHYS-FULL-DETAILED"][[f"{truss_prefix}_{p}" for p in PARAMS]].copy()
    truss.columns = PARAMS
    truss = truss.astype(float)

    rows = []
    for param in PARAMS:
        c = chess[param].to_numpy(dtype=float)
        t = truss[param].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci_diff(c, t, seed=100 + PARAMS.index(param))
        pooled_sd = np.sqrt(((c.size - 1) * c.var(ddof=1) + (t.size - 1) * t.var(ddof=1)) / (c.size + t.size - 2))
        cohens_d = (t.mean() - c.mean()) / pooled_sd
        tost = tost_equivalence(c, t)
        bf = bic_bayes_factor_equal_vs_independent(c, t)
        rows.append({
            "parameter": param,
            "channel": channel,
            "chess_mean": float(c.mean()),
            "chess_std": float(c.std(ddof=1)),
            "truss_mean": float(t.mean()),
            "truss_std": float(t.std(ddof=1)),
            "mean_diff_truss_minus_chess": float(t.mean() - c.mean()),
            "pooled_sd": float(pooled_sd),
            "cohens_d_truss_minus_chess": float(cohens_d),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "permutation_p": permutation_pvalue_1d(c, t, seed=200 + PARAMS.index(param)),
            **tost,
            **bf,
        })
    param_results = pd.DataFrame(rows)

    vector_stat, vector_p = permutation_pvalue_vector(chess[PARAMS].to_numpy(dtype=float), truss[PARAMS].to_numpy(dtype=float))
    global_results = pd.DataFrame([{
        "channel": channel,
        "test": "multivariate_standardized_mean_vector_distance",
        "parameters": ",".join(PARAMS),
        "statistic": vector_stat,
        "permutation_p": vector_p,
        "n_chess": int(chess.shape[0]),
        "n_truss": int(truss.shape[0]),
    }])

    return param_results, global_results


def main() -> None:
    self_results, self_global = compare_channel("self", CHESS_SELF_PATH, "self")
    ai_results, ai_global = compare_channel("AI", CHESS_AI_PATH, "AI")

    self_results.to_csv(OUT_DIR / "chess_truss_self_param_comparison.csv", index=False)
    self_results.to_csv(OUT_DIR / "chess_truss_self_param_tost_bayes.csv", index=False)
    self_global.to_csv(OUT_DIR / "chess_truss_self_param_global_test.csv", index=False)
    ai_results.to_csv(OUT_DIR / "chess_truss_ai_param_comparison.csv", index=False)
    ai_results.to_csv(OUT_DIR / "chess_truss_ai_param_tost_bayes.csv", index=False)
    ai_global.to_csv(OUT_DIR / "chess_truss_ai_param_global_test.csv", index=False)

    param_results = pd.concat([self_results, ai_results], ignore_index=True)
    global_results = pd.concat([self_global, ai_global], ignore_index=True)
    param_results.to_csv(OUT_DIR / "chess_truss_all_channel_param_comparison.csv", index=False)
    global_results.to_csv(OUT_DIR / "chess_truss_all_channel_param_global_test.csv", index=False)

    print("\nChess vs truss PHYS-FULL parameter comparison")
    print(param_results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nGlobal same-distribution permutation test")
    print(global_results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()

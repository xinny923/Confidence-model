from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data_folder_output"
RECON_PATH = OUT_DIR / "data_folder_reconstructed_trials.csv"


def load_feed2_by_condition() -> dict[int, np.ndarray]:
    mat = loadmat(ROOT / "feed2_data.mat", squeeze_me=False, struct_as_record=False)
    cell = np.asarray(mat["feed2_data"])
    return {cond: np.asarray(item, dtype=float) for cond, item in enumerate(cell.ravel(order="C"), start=1)}


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return float(center - half), float(center + half)


def positive_feedback_column(df: pd.DataFrame) -> pd.Series:
    feed2_by_cond = load_feed2_by_condition()
    values = []
    for row in df.itertuples(index=False):
        feed2 = feed2_by_cond[int(row.condition)][int(row.condition_row), int(row.trial) - 1]
        values.append(feed2 > 0)
    return pd.Series(values, index=df.index, dtype=bool)


def rate_row(label: str, numerator: int, denominator: int) -> dict[str, float | int | str]:
    lo, hi = wilson_ci(numerator, denominator)
    return {
        "rate": label,
        "numerator": numerator,
        "denominator": denominator,
        "percent": 100 * numerator / denominator,
        "ci_low_percent": 100 * lo,
        "ci_high_percent": 100 * hi,
    }


def main() -> None:
    df = pd.read_csv(RECON_PATH)
    usable = df[
        df["has_move1"]
        & df["has_move2"]
        & df["has_suggestion"]
        & (df["reconstructed_action"] != "missing")
    ].copy()
    usable["alignment_status"] = np.where(usable["align"] == 1, "Aligned", "Non-aligned")
    usable["positive_feedback"] = positive_feedback_column(usable)

    action_order = ["accept_ai", "modify", "reject_ai"]
    table = pd.crosstab(usable["alignment_status"], usable["reconstructed_action"]).reindex(
        index=["Aligned", "Non-aligned"],
        columns=action_order,
        fill_value=0,
    )
    table = table.rename(columns={"accept_ai": "Accept", "modify": "Modify", "reject_ai": "Reject"})
    table["Total"] = table.sum(axis=1)
    total_row = table.sum(axis=0)
    total_row.name = "Total"
    table = pd.concat([table, total_row.to_frame().T])

    chi2, chi2_p, dof, expected = stats.chi2_contingency(table.loc[["Aligned", "Non-aligned"], ["Accept", "Modify", "Reject"]])
    n = int(table.loc["Total", "Total"])
    cramers_v = float(np.sqrt(chi2 / (n * (min(2, 3) - 1))))

    # 2x2 accept vs not-accept contrast.
    accept_2x2 = np.array([
        [int(table.loc["Aligned", "Accept"]), int(table.loc["Aligned", "Total"] - table.loc["Aligned", "Accept"])],
        [int(table.loc["Non-aligned", "Accept"]), int(table.loc["Non-aligned", "Total"] - table.loc["Non-aligned", "Accept"])],
    ])
    accept_or, accept_fisher_p = stats.fisher_exact(accept_2x2)

    rates = pd.DataFrame([
        rate_row("aligned_accept_rate", int(table.loc["Aligned", "Accept"]), int(table.loc["Aligned", "Total"])),
        rate_row("nonaligned_accept_rate", int(table.loc["Non-aligned", "Accept"]), int(table.loc["Non-aligned", "Total"])),
        rate_row("aligned_modify_rate", int(table.loc["Aligned", "Modify"]), int(table.loc["Aligned", "Total"])),
        rate_row("nonaligned_modify_rate", int(table.loc["Non-aligned", "Modify"]), int(table.loc["Non-aligned", "Total"])),
        rate_row("aligned_reject_rate", int(table.loc["Aligned", "Reject"]), int(table.loc["Aligned", "Total"])),
        rate_row("nonaligned_reject_rate", int(table.loc["Non-aligned", "Reject"]), int(table.loc["Non-aligned", "Total"])),
    ])

    outcome_rows = []
    for status in ["Aligned", "Non-aligned"]:
        for action in action_order:
            sub = usable[(usable["alignment_status"] == status) & (usable["reconstructed_action"] == action)]
            pos = int(sub["positive_feedback"].sum())
            total = int(sub.shape[0])
            lo, hi = wilson_ci(pos, total)
            outcome_rows.append({
                "alignment_status": status,
                "action": {"accept_ai": "Accept", "modify": "Modify", "reject_ai": "Reject"}[action],
                "positive": pos,
                "total": total,
                "positive_percent": 100 * pos / total if total else np.nan,
                "ci_low_percent": 100 * lo,
                "ci_high_percent": 100 * hi,
            })
    outcome = pd.DataFrame(outcome_rows)

    stats_df = pd.DataFrame([
        {"test": "chi_square_2x3", "statistic": chi2, "p_value": chi2_p, "df": dof, "effect": "cramers_v", "effect_value": cramers_v},
        {"test": "fisher_accept_vs_nonaccept", "statistic": accept_or, "p_value": accept_fisher_p, "df": np.nan, "effect": "odds_ratio", "effect_value": accept_or},
    ])

    table.to_csv(OUT_DIR / "alignment_reliance_contingency.csv")
    rates.to_csv(OUT_DIR / "alignment_reliance_rates.csv", index=False)
    outcome.to_csv(OUT_DIR / "alignment_reliance_positive_feedback.csv", index=False)
    stats_df.to_csv(OUT_DIR / "alignment_reliance_tests.csv", index=False)

    print("\nTRUSS RELIANCE x ALIGNMENT BREAKDOWN")
    print(table.to_string())
    print("\nTests")
    print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nRates")
    for row in rates.itertuples(index=False):
        print(f"{row.rate}: {row.numerator}/{row.denominator} = {row.percent:.1f}% [{row.ci_low_percent:.1f}, {row.ci_high_percent:.1f}]")
    print("\nPositive feedback within alignment/action cells")
    print(outcome.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()

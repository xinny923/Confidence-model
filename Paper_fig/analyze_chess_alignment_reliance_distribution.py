from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

PAPER_DIR = Path(__file__).resolve().parent
OUT_DIR = PAPER_DIR / "manu_fig"
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

import full_model as full  # noqa: E402


ACTION_LABELS = {0: "Accept", 1: "Reject", 2: "Modify"}
ACTION_ORDER = ["Accept", "Modify", "Reject"]


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return float(center - half), float(center + half)


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts, _ = full.load_participants(full.Config.DATA_DIR)
    mats = full.build_matrices(parts)
    records = []
    for p_idx, participant in enumerate(parts):
        for t, rec in enumerate(participant.trial_records):
            records.append({
                "participant": participant.pid,
                "condition": participant.condition,
                "trial": t + 1,
                "align": int(mats["align_tensor"][p_idx, t]),
                "alignment_status": "Aligned" if mats["align_tensor"][p_idx, t] == 1 else "Non-aligned",
                "action_code": int(rec["action"]),
                "action": ACTION_LABELS[int(rec["action"])],
                "feedback": float(rec["feedback"]),
                "positive_feedback": float(rec["feedback"]) > 0,
            })
    df = pd.DataFrame(records)

    table = pd.crosstab(df["alignment_status"], df["action"]).reindex(
        index=["Aligned", "Non-aligned"],
        columns=ACTION_ORDER,
        fill_value=0,
    )
    table["Total"] = table.sum(axis=1)
    total_row = table.sum(axis=0)
    total_row.name = "Total"
    table = pd.concat([table, total_row.to_frame().T])

    chi2, chi2_p, dof, _ = stats.chi2_contingency(table.loc[["Aligned", "Non-aligned"], ACTION_ORDER])
    n = int(table.loc["Total", "Total"])
    cramers_v = float(np.sqrt(chi2 / (n * (min(2, 3) - 1))))

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
        for action in ACTION_ORDER:
            sub = df[(df["alignment_status"] == status) & (df["action"] == action)]
            pos = int(sub["positive_feedback"].sum())
            total = int(sub.shape[0])
            lo, hi = wilson_ci(pos, total)
            outcome_rows.append({
                "alignment_status": status,
                "action": action,
                "positive": pos,
                "total": total,
                "positive_percent": 100 * pos / total if total else np.nan,
                "ci_low_percent": 100 * lo,
                "ci_high_percent": 100 * hi,
            })
    outcome = pd.DataFrame(outcome_rows)
    tests = pd.DataFrame([
        {"test": "chi_square_2x3", "statistic": chi2, "p_value": chi2_p, "df": dof, "effect": "cramers_v", "effect_value": cramers_v},
        {"test": "fisher_accept_vs_nonaccept", "statistic": accept_or, "p_value": accept_fisher_p, "df": np.nan, "effect": "odds_ratio", "effect_value": accept_or},
    ])

    table.to_csv(OUT_DIR / "chess_alignment_reliance_contingency.csv")
    rates.to_csv(OUT_DIR / "chess_alignment_reliance_rates.csv", index=False)
    outcome.to_csv(OUT_DIR / "chess_alignment_reliance_positive_feedback.csv", index=False)
    tests.to_csv(OUT_DIR / "chess_alignment_reliance_tests.csv", index=False)

    print("\nCHESS RELIANCE x ALIGNMENT BREAKDOWN")
    print(table.to_string())
    print("\nTests")
    print(tests.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nRates")
    for row in rates.itertuples(index=False):
        print(f"{row.rate}: {row.numerator}/{row.denominator} = {row.percent:.1f}% [{row.ci_low_percent:.1f}, {row.ci_high_percent:.1f}]")
    print("\nPositive feedback within alignment/action cells")
    print(outcome.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()

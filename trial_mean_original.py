"""
Trial-level mean residuals for the original 4-term model (table1_model_params.csv).
Outputs to output_trial_mean_orig/:
  - trial_mean_residuals.csv (cond, track, trial, mean_obs, mean_pred, resid, abs_resid)
  - summary.txt (Top5 worst trials per condition for AI/Self)
  - actions_cond1.csv / actions_cond2.csv (action percentages per trial)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import original_common as oc


def main() -> None:
    out_dir = Path("output_trial_mean_orig")
    out_dir.mkdir(parents=True, exist_ok=True)

    params = pd.read_csv(oc.ConfigOrig.PARAMS_PATH)
    ai_p = params[params["model"] == "aiconf"].iloc[0][oc.ConfigOrig.PARAMETER_NAMES].values
    self_p = params[params["model"] == "selfconf"].iloc[0][oc.ConfigOrig.PARAMETER_NAMES].values

    parts, trial_df = oc.load_all_participants(oc.ConfigOrig.DATA_DIR)
    matrices = oc.build_analysis_matrices(parts)

    pred_ai = oc.compute_model_predictions(ai_p, matrices["ai_conf_matrix"], matrices["e_tensor"])
    pred_self = oc.compute_model_predictions(self_p, matrices["self_conf_matrix"], matrices["e_tensor"])

    records = []
    for cond in [1, 2]:
        idx = np.where(matrices["condition_array"] == cond)[0]
        mean_obs_ai = np.nanmean(matrices["ai_conf_matrix"][idx], axis=0)
        mean_pred_ai = np.nanmean(pred_ai[idx], axis=0)
        mean_obs_self = np.nanmean(matrices["self_conf_matrix"][idx], axis=0)
        mean_pred_self = np.nanmean(pred_self[idx], axis=0)
        for t in range(1, oc.ConfigOrig.NUM_TRIALS + 1):
            records.append({
                "condition": cond,
                "track": "ai",
                "trial": t,
                "mean_obs": mean_obs_ai[t],
                "mean_pred": mean_pred_ai[t],
                "resid": mean_obs_ai[t] - mean_pred_ai[t],
                "abs_resid": abs(mean_obs_ai[t] - mean_pred_ai[t]),
            })
            records.append({
                "condition": cond,
                "track": "self",
                "trial": t,
                "mean_obs": mean_obs_self[t],
                "mean_pred": mean_pred_self[t],
                "resid": mean_obs_self[t] - mean_pred_self[t],
                "abs_resid": abs(mean_obs_self[t] - mean_pred_self[t]),
            })

    df_res = pd.DataFrame(records)
    df_res.to_csv(out_dir / "trial_mean_residuals.csv", index=False)

    summary = []
    for track in ["ai", "self"]:
        summary.append(f"\nTop5 trials by abs residual ({track}):")
        for cond in [1, 2]:
            sub = df_res[(df_res["condition"] == cond) & (df_res["track"] == track)]
            top5 = sub.sort_values("abs_resid", ascending=False).head(5)[["trial", "abs_resid"]]
            summary.append(
                f"  cond {cond}: " +
                ", ".join([f"{int(r.trial)}({r.abs_resid:.3f})" for r in top5.itertuples()])
            )

    trial_df["trial"] = trial_df["trial"].astype(int)
    for cond in [1, 2]:
        sub = trial_df[trial_df["condition"] == cond]
        acts = sub.groupby("trial")["action_label"].value_counts(normalize=True).unstack().fillna(0) * 100
        acts.round(1).to_csv(out_dir / f"actions_cond{cond}.csv")

    (out_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("Saved trial_mean_residuals.csv and summary.txt in", out_dir)


if __name__ == "__main__":
    main()

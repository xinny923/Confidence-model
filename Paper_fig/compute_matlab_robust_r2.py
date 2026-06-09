from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "Code"
OUT_DIR = ROOT / "Paper_fig" / "full_output" / "matlab_robust_r2"


def mat(name: str, key: str) -> np.ndarray:
    return np.asarray(loadmat(CODE_DIR / name)[key], dtype=float)


def predict_matlab(
    x: np.ndarray,
    observed: np.ndarray,
    e_tensor: np.ndarray,
    baseline_b: np.ndarray | None = None,
) -> np.ndarray:
    """Replicate Code/modelPlot.m and Code/modelPlot_self.m prediction loops."""
    x = np.asarray(x, dtype=float).ravel()
    c = observed[:, 0].copy()
    b = observed[:, 0].copy() if baseline_b is None else np.asarray(baseline_b, dtype=float).ravel().copy()
    pred = [c.copy()]
    a = None
    for i in range(30):
        if i == 0:
            a = observed[:, i].copy()
        else:
            a = x[7] * pred[i - 1] + (1.0 - x[7]) * a
        e = (
            x[3] * e_tensor[:, i, 0]
            + x[4] * e_tensor[:, i, 1]
            + x[5] * e_tensor[:, i, 2]
            + x[6] * e_tensor[:, i, 3]
        )
        c = c + x[0] * (e - c) + x[1] * (a - c) + x[2] * (b - c)
        pred.append(c.copy())
    return np.column_stack(pred)


def matlab_curve_metrics(
    x: np.ndarray,
    cond_data: list[np.ndarray],
    cond_e: list[np.ndarray],
    cond_b: list[np.ndarray | None],
    cond_indices: list[np.ndarray] | None = None,
) -> dict[str, float]:
    ssres = 0.0
    sstot = 0.0
    mse_parts = []
    for cond_i, (data, e_tensor, b_vec) in enumerate(zip(cond_data, cond_e, cond_b)):
        if cond_indices is not None:
            idx = cond_indices[cond_i]
            data = data[idx]
            e_tensor = e_tensor[idx]
            if b_vec is not None:
                b_vec = b_vec[idx]
        pred = predict_matlab(x, data, e_tensor, baseline_b=b_vec)
        mean_obs = np.nanmean(data, axis=0)
        mean_pred = np.nanmean(pred, axis=0)
        resid = mean_pred[1:] - mean_obs[1:]
        ssres += float(np.nansum(resid**2))
        sstot += float(np.nansum((mean_obs[1:] - np.nanmean(mean_obs[1:])) ** 2))
        mse_parts.append(float(np.nanmean(resid**2)))
    r2 = 1.0 - ssres / sstot
    adj_r2 = 1.0 - (99.0 / 92.0) * (ssres / sstot)
    return {
        "mse": ssres / 60.0,
        "r2": r2,
        "adj_r2": adj_r2,
        "cond1_mse": mse_parts[0],
        "cond2_mse": mse_parts[1],
    }


def split_matlab_chosen(chosen_1based: np.ndarray) -> list[np.ndarray]:
    """Map robust chosen ids from merged 100-row MATLAB matrix to condition-local rows."""
    chosen0 = np.asarray(chosen_1based, dtype=int).ravel() - 1
    cond1 = chosen0[chosen0 < 50]
    cond2 = chosen0[chosen0 >= 50] - 50
    return [np.sort(cond1), np.sort(cond2)]


def summarize(df: pd.DataFrame, label: str) -> dict[str, float | str]:
    return {
        "track": label,
        "n": int(df.shape[0]),
        "mse_mean": float(df["mse"].mean()),
        "mse_std": float(df["mse"].std(ddof=1)),
        "r2_mean": float(df["r2"].mean()),
        "r2_std": float(df["r2"].std(ddof=1)),
        "adj_r2_mean": float(df["adj_r2"].mean()),
        "adj_r2_std": float(df["adj_r2"].std(ddof=1)),
        "adj_r2_min": float(df["adj_r2"].min()),
        "adj_r2_max": float(df["adj_r2"].max()),
    }


def run_track(track: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if track == "aiconf":
        cond_data = [mat("C_data1.mat", "C_data"), mat("C_data2.mat", "C_data")]
        cond_b = [mat("B0_data1.mat", "B0_data").ravel(), mat("B0_data2.mat", "B0_data").ravel()]
        robust_params = np.loadtxt(CODE_DIR / "x_robust.csv", delimiter=",")
        robust_chosen = np.loadtxt(CODE_DIR / "data_robust.csv", delimiter=",")
        full_x = mat("x.mat", "x").ravel()
    elif track == "selfconf":
        cond_data = [mat("selfC_data1.mat", "selfC_data"), mat("selfC_data2.mat", "selfC_data")]
        cond_b = [None, None]
        robust_params = np.loadtxt(CODE_DIR / "x_robust_self.csv", delimiter=",")
        robust_chosen = np.loadtxt(CODE_DIR / "data_robust_self.csv", delimiter=",")
        full_x = mat("selfx.mat", "x").ravel()
    else:
        raise ValueError(track)

    cond_e = [mat("e_data1.mat", "e_data"), mat("e_data2.mat", "e_data")]

    rows = []
    for scope in ["full_data_curve", "chosen_subset_curve"]:
        for iteration, x in enumerate(robust_params, start=1):
            cond_indices = None
            if scope == "chosen_subset_curve":
                cond_indices = split_matlab_chosen(robust_chosen[iteration - 1])
            metrics = matlab_curve_metrics(x, cond_data, cond_e, cond_b, cond_indices)
            row = {"track": track, "scope": scope, "iteration": iteration}
            row.update(metrics)
            rows.append(row)

    full_metrics = []
    for scope in ["full_data_curve"]:
        metrics = matlab_curve_metrics(full_x, cond_data, cond_e, cond_b)
        row = {"track": track, "scope": "original_x_full_data_curve", "iteration": 0}
        row.update(metrics)
        full_metrics.append(row)

    return pd.DataFrame(rows), pd.DataFrame(full_metrics)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    full_rows = []
    summaries = []
    for track in ["aiconf", "selfconf"]:
        robust, full = run_track(track)
        all_rows.append(robust)
        full_rows.append(full)
        for scope, group in robust.groupby("scope"):
            summaries.append(summarize(group, f"{track}:{scope}"))
    robust_df = pd.concat(all_rows, ignore_index=True)
    full_df = pd.concat(full_rows, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    robust_df.to_csv(OUT_DIR / "matlab_robust_r2_raw.csv", index=False)
    full_df.to_csv(OUT_DIR / "matlab_original_x_r2.csv", index=False)
    summary_df.to_csv(OUT_DIR / "matlab_robust_r2_summary.csv", index=False)

    print("\nOriginal MATLAB x on full condition mean curves")
    print(full_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nRobust MATLAB params R2 summary")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()

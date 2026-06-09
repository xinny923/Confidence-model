from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({
    "font.size": 6,
    "axes.labelsize": 6,
    "axes.titlesize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "svg.fonttype": "none",
})

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "Paper_fig"
TRUSS_DIR = ROOT / "truss_data"
OUT_DIR = PAPER_DIR / "manu_fig"

D_VALUES = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0], dtype=float)
COLORS = {
    "data": "#8c8c8c",
    "baseline": "#1f2a6d",
    "full": "#b20d0d",
    "ai": "#4e79a7",
    "self": "#e15759",
    "shade": "#d8d8d8",
    "grid": "#b0b0b0",
}


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


def save_figure(fig: plt.Figure, stem: str, aliases: tuple[str, ...] = ()) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in (stem, *aliases):
        png = OUT_DIR / f"{name}.png"
        pdf = OUT_DIR / f"{name}.pdf"
        svg = OUT_DIR / f"{name}.svg"
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        print(f"Saved {png}")
        print(f"Saved {pdf}")
        print(f"Saved {svg}")
    plt.close(fig)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]))


def load_chess_predictions() -> dict[str, np.ndarray]:
    comp = load_module("manu_chess_component", PAPER_DIR / "chess_participant_robust_comparison.py")
    og = comp.og
    full = comp.full

    base_parts, _ = og.load_all_participants(og.Config.DATA_DIR)
    base_mats = og.build_analysis_matrices(base_parts)
    full_parts, _ = full.load_participants(full.Config.DATA_DIR)
    full_mats = full.build_matrices(full_parts)

    base_ai_fit = comp.fit_og_track("aiconf", base_mats["ai_conf_matrix"], base_mats["e_tensor"], base_mats["condition_array"])
    base_self_fit = comp.fit_og_track("selfconf", base_mats["self_conf_matrix"], base_mats["e_tensor"], base_mats["condition_array"])
    full_variant = next(v for v in comp.VARIANTS if v.name == "PHYS-FULL")
    full_ai_fit = comp.fit_variant(
        "aiconf",
        full_variant,
        full_mats["ai_conf_matrix"],
        full_mats["e_tensor"],
        full_mats["align_tensor"],
        full_mats["condition_array"],
        False,
    )
    full_self_fit = comp.fit_variant(
        "selfconf",
        full_variant,
        full_mats["self_conf_matrix"],
        full_mats["e_tensor"],
        full_mats["align_tensor"],
        full_mats["condition_array"],
        True,
    )

    actions = np.array([[r["action"] for r in p.trial_records] for p in full_parts], dtype=int)
    align = full_mats["align_tensor"].astype(bool)
    return {
        "component_module": comp,
        "full_variant": full_variant,
        "full_ai_param_names": full_ai_fit["param_names"],
        "full_self_param_names": full_self_fit["param_names"],
        "condition": base_mats["condition_array"],
        "ai": base_mats["ai_conf_matrix"],
        "self": base_mats["self_conf_matrix"],
        "e_tensor": full_mats["e_tensor"],
        "align": align,
        "actions": actions,
        "base_ai": base_ai_fit["pred"],
        "base_self": base_self_fit["pred"],
        "full_ai": full_ai_fit["pred"],
        "full_self": full_self_fit["pred"],
        "full_ai_params": full_ai_fit["params"],
        "full_self_params": full_self_fit["params"],
    }


def apply_d_to_modify_trials(e_tensor: np.ndarray, actions: np.ndarray, d_value: float) -> np.ndarray:
    """Apply manuscript partial-weight parameter only to modify trials.

    The Python evidence order is accept+, reject+, accept-, reject-. In the
    manuscript equation, d weights the accept/AI-side channels and (1-d)
    weights the reject/self-side channels.
    """
    weighted = e_tensor.copy()
    modify = actions == 2
    positive = modify & (np.nansum(e_tensor[:, :, :2], axis=2) > 0)
    negative = modify & (np.nansum(e_tensor[:, :, 2:], axis=2) > 0)
    weighted[positive, 0] = d_value
    weighted[positive, 1] = 1.0 - d_value
    weighted[positive, 2] = 0.0
    weighted[positive, 3] = 0.0
    weighted[negative, 0] = 0.0
    weighted[negative, 1] = 0.0
    weighted[negative, 2] = d_value
    weighted[negative, 3] = 1.0 - d_value
    return weighted


def figure3_d_sweep(data: dict[str, np.ndarray]) -> pd.DataFrame:
    comp = data["component_module"]
    rows = []
    modify = data["actions"] == 2
    for d_value in D_VALUES:
        e_tensor = apply_d_to_modify_trials(data["e_tensor"], data["actions"], d_value)
        ai_pred = comp.predict_variant(
            data["full_ai_params"],
            data["full_ai_param_names"],
            data["full_variant"],
            data["ai"],
            e_tensor,
            data["align"],
            False,
        )
        self_pred = comp.predict_variant(
            data["full_self_params"],
            data["full_self_param_names"],
            data["full_variant"],
            data["self"],
            e_tensor,
            data["align"],
            True,
        )
        rows.append({
            "d": float(d_value),
            "ai_mse": float(np.nanmean((data["ai"][:, 1:][modify] - ai_pred[:, 1:][modify]) ** 2)),
            "self_mse": float(np.nanmean((data["self"][:, 1:][modify] - self_pred[:, 1:][modify]) ** 2)),
            "n_modify_points": int(np.isfinite(data["ai"][:, 1:][modify]).sum()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "fig3_chess_d_sweep.csv", index=False)
    df.to_csv(OUT_DIR / "fig2_chess_rho_sweep.csv", index=False)

    plt.rcParams.update({"font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6})
    fig, axes = plt.subplots(2, 1, figsize=(3.8, 2.25), sharex=True)
    panels = [
        ("ai_mse", "AI MSE", COLORS["baseline"], "(a)"),
        ("self_mse", "Self MSE", COLORS["full"], "(b)"),
    ]
    for ax, (col, ylabel, color, label) in zip(axes, panels):
        ax.axvspan(0.2, 0.8, color=COLORS["shade"], alpha=0.45, zorder=0)
        ax.plot(df["d"], df[col], color=color, marker="o", markersize=4.0, linewidth=1.6)
        pad = max((df[col].max() - df[col].min()) * 0.18, 1e-5)
        ax.set_ylim(df[col].min() - pad, df[col].max() + pad)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(alpha=0.18, color=COLORS["grid"], linewidth=0.6)
        ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=6, fontweight="bold", clip_on=False)
    axes[-1].set_xlabel(r"$\rho$")
    axes[-1].set_xticks(D_VALUES)
    fig.tight_layout()
    save_figure(fig, "fig3_chess_d_sweep", aliases=("fig2_chess_rho_sweep", "figure/fig2_rho"))
    return df


def action_gap_rows(data: dict[str, np.ndarray]) -> pd.DataFrame:
    labels = {0: "accept", 1: "reject", 2: "modify"}
    tracks = {
        "Experiment Data": (data["ai"], data["self"]),
        "Baseline Model": (data["base_ai"], data["base_self"]),
        "Reformulated Model": (data["full_ai"], data["full_self"]),
    }
    rows = []
    for model, (ai, selfc) in tracks.items():
        gap = np.abs(np.diff(ai, axis=1)) - np.abs(np.diff(selfc, axis=1))
        for code, action in labels.items():
            vals = gap[data["actions"] == code]
            low, high = bootstrap_ci(vals, seed=100 + code)
            rows.append({
                "model": model,
                "action": action,
                "gap": float(np.nanmean(vals)),
                "ci_low": low,
                "ci_high": high,
                "n": int(np.isfinite(vals).sum()),
            })
    return pd.DataFrame(rows)


def figure4_action_gap(data: dict[str, np.ndarray]) -> pd.DataFrame:
    df = action_gap_rows(data)
    df.to_csv(OUT_DIR / "fig4_chess_action_gap.csv", index=False)
    df.to_csv(OUT_DIR / "fig3_chess_action_gap.csv", index=False)

    fig4_font = 10
    actions = ["accept", "modify", "reject"]
    offsets = {"Experiment Data": -0.18, "Baseline Model": 0.0, "Reformulated Model": 0.18}
    colors = {"Experiment Data": COLORS["data"], "Baseline Model": COLORS["baseline"], "Reformulated Model": COLORS["full"]}
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    x = np.arange(len(actions))
    for model in ["Experiment Data", "Baseline Model", "Reformulated Model"]:
        sub = df[df["model"] == model].set_index("action").loc[actions].reset_index()
        xpos = x + offsets[model]
        yerr = np.vstack([sub["gap"] - sub["ci_low"], sub["ci_high"] - sub["gap"]])
        ax.errorbar(xpos, sub["gap"], yerr=yerr, fmt="o", color=colors[model], capsize=3, markersize=4, label=model)
    ax.axhline(0, color="#666666", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(actions)
    ax.set_ylabel("d(a)=|delta C AI|-|delta C self|")
    ax.tick_params(labelsize=fig4_font)
    ax.xaxis.label.set_size(fig4_font)
    ax.yaxis.label.set_size(fig4_font)
    ax.grid(axis="y", alpha=0.16)
    ax.legend(loc="upper right", fontsize=fig4_font, frameon=True)
    fig.tight_layout()
    save_figure(fig, "fig4_chess_action_gap", aliases=("fig3_chess_action_gap", "figure/fig3_mod"))
    return df


def align_summary(data: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    align = data["align"]
    observed = {"AI channel": data["ai"], "Self channel": data["self"]}
    models = {
        "Baseline Model": {"AI channel": data["base_ai"], "Self channel": data["base_self"]},
        "Reformulated Model": {"AI channel": data["full_ai"], "Self channel": data["full_self"]},
    }
    tracks = {"Experiment Data": observed, **models}
    box_rows = []
    trial_rows = []
    for model, predictions in models.items():
        for channel, pred in predictions.items():
            for trial in range(30):
                mask = align[:, trial]
                if not mask.any():
                    continue
                obs_mean = float(np.nanmean(observed[channel][mask, trial + 1]))
                pred_mean = float(np.nanmean(pred[mask, trial + 1]))
                box_rows.append({
                    "model": model,
                    "channel": channel,
                    "trial": trial + 1,
                    "abs_error": abs(obs_mean - pred_mean),
                    "align_count": int(mask.sum()),
                })

    for model, channels in tracks.items():
        ai_abs = np.abs(np.diff(channels["AI channel"], axis=1))
        for trial in range(30):
            vals = ai_abs[:, trial][align[:, trial]]
            low, high = bootstrap_ci(vals, seed=300 + trial)
            trial_rows.append({
                "model": model,
                "trial": trial + 1,
                "mean_abs_delta": float(np.nanmean(vals)) if vals.size else np.nan,
                "ci_low": low,
                "ci_high": high,
                "align_count": int(align[:, trial].sum()),
            })
    error_df = pd.DataFrame(box_rows)
    improvement_rows = []
    for channel in ["AI channel", "Self channel"]:
        base = error_df[(error_df["model"] == "Baseline Model") & (error_df["channel"] == channel)].set_index("trial")
        full = error_df[(error_df["model"] == "Reformulated Model") & (error_df["channel"] == channel)].set_index("trial")
        common_trials = base.index.intersection(full.index)
        for trial in common_trials:
            exp = observed[channel]
            mask = align[:, int(trial) - 1]
            empirical_delta = float(np.nanmean(np.abs(np.diff(exp, axis=1)[:, int(trial) - 1][mask])))
            improvement_rows.append({
                "channel": channel,
                "trial": int(trial),
                "align_count": int(base.loc[trial, "align_count"]),
                "baseline_abs_error": float(base.loc[trial, "abs_error"]),
                "full_abs_error": float(full.loc[trial, "abs_error"]),
                "improvement": float(base.loc[trial, "abs_error"] - full.loc[trial, "abs_error"]),
                "empirical_abs_delta": empirical_delta,
            })
    return error_df, pd.DataFrame(trial_rows), pd.DataFrame(improvement_rows)


def alignment_case_improvement(data: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    align = data["align"].astype(bool)
    specs = [
        ("AI channel", data["ai"], data["base_ai"], data["full_ai"]),
        ("Self channel", data["self"], data["base_self"], data["full_self"]),
    ]
    align_counts = align.sum(axis=0).astype(int)
    for channel, observed, baseline, reformulated in specs:
        for trial in range(1, observed.shape[1]):
            mask = align[:, trial - 1]
            for participant_index in np.where(mask)[0]:
                base_error = abs(observed[participant_index, trial] - baseline[participant_index, trial])
                full_error = abs(observed[participant_index, trial] - reformulated[participant_index, trial])
                rows.append({
                    "channel": channel,
                    "participant_index": int(participant_index),
                    "trial": int(trial),
                    "align_count": int(align_counts[trial - 1]),
                    "improvement": float(base_error - full_error),
                    "empirical_abs_delta": float(abs(observed[participant_index, trial] - observed[participant_index, trial - 1])),
                })
    return pd.DataFrame(rows)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")


def weighted_bootstrap_ci(values: np.ndarray, weights: np.ndarray, seed: int) -> tuple[float, float]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    vals = values[valid]
    w = weights[valid]
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    probs = w / w.sum()
    samples = rng.choice(np.arange(vals.size), size=(2000, vals.size), replace=True, p=probs)
    boot = np.array([np.average(vals[idx], weights=w[idx]) for idx in samples])
    return tuple(np.percentile(boot, [2.5, 97.5]))


def figure5_alignment(data: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    error_df, trial_df, improvement_df = align_summary(data)
    error_df.to_csv(OUT_DIR / "fig5_chess_align_error.csv", index=False)
    trial_df.to_csv(OUT_DIR / "fig5_chess_align_trial_delta.csv", index=False)
    improvement_df.to_csv(OUT_DIR / "fig5_chess_align_improvement.csv", index=False)
    error_df.to_csv(OUT_DIR / "fig4_chess_align_box.csv", index=False)
    trial_df.to_csv(OUT_DIR / "fig4_chess_align_trial.csv", index=False)

    fig5_font = 8
    fig = plt.figure(figsize=(3.6, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.34)
    ax_box = fig.add_subplot(gs[0, 0])
    ax_line = fig.add_subplot(gs[1, 0])

    channels = ["AI channel", "Self channel"]
    models = ["Baseline Model", "Reformulated Model"]
    positions = [0.76, 1.02, 1.36, 1.62]
    box_data = [
        np.repeat(
            error_df[(error_df["channel"] == channel) & (error_df["model"] == model)]["abs_error"].to_numpy(dtype=float),
            error_df[(error_df["channel"] == channel) & (error_df["model"] == model)]["align_count"].to_numpy(dtype=int),
        )
        for channel in channels
        for model in models
    ]
    plot_colors = [COLORS["baseline"], COLORS["full"], COLORS["baseline"], COLORS["full"]]
    bp = ax_box.boxplot(box_data, positions=positions, widths=0.18, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.0)
    for whisker in bp["whiskers"]:
        whisker.set_color("#777777")
        whisker.set_linewidth(0.9)
    for cap in bp["caps"]:
        cap.set_color("#777777")
        cap.set_linewidth(0.9)
    for median in bp["medians"]:
        median.set_color("#333333")
        median.set_linewidth(1.0)
    ax_box.set_xticks([0.89, 1.49])
    ax_box.set_xticklabels(channels)
    ax_box.set_ylabel("|error|")
    ax_box.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax_box.tick_params(labelsize=fig5_font)
    ax_box.xaxis.label.set_size(fig5_font)
    ax_box.yaxis.label.set_size(fig5_font)
    ax_box.yaxis.labelpad = 1
    ax_box.text(-0.075, 1.08, "(a)", transform=ax_box.transAxes, fontsize=fig5_font, fontweight="bold", clip_on=False)
    ax_box.legend(
        [bp["boxes"][0], bp["boxes"][1]],
        models,
        loc="upper right",
        fontsize=fig5_font,
        frameon=True,
        handlelength=1.4,
        borderpad=0.25,
        labelspacing=0.2,
    )
    ax_box.grid(axis="y", alpha=0.16)

    ax_count = ax_line.twinx()
    counts = trial_df[trial_df["model"] == "Experiment Data"].sort_values("trial")
    ax_count.bar(counts["trial"], counts["align_count"], color="#d9d9d9", width=0.8, alpha=0.75, zorder=0)
    ax_count.set_ylabel("Alignment case count")
    ax_count.tick_params(labelsize=fig5_font)
    ax_count.yaxis.label.set_size(fig5_font)
    ax_count.yaxis.labelpad = 1
    ax_count.set_ylim(0, max(5, counts["align_count"].max() * 1.25))

    style = {
        "Experiment Data": (COLORS["data"], "o"),
        "Baseline Model": (COLORS["baseline"], "s"),
        "Reformulated Model": (COLORS["full"], "^"),
    }
    for model, (color, marker) in style.items():
        sub = trial_df[trial_df["model"] == model].sort_values("trial")
        yerr = np.vstack([sub["mean_abs_delta"] - sub["ci_low"], sub["ci_high"] - sub["mean_abs_delta"]])
        ax_line.errorbar(sub["trial"], sub["mean_abs_delta"], yerr=yerr, fmt=marker, color=color, markersize=3, capsize=2, linewidth=1.0, label=model, zorder=3)
    ax_line.set_xlabel("Trial")
    ax_line.set_ylabel(r"mean $|\Delta C_{AI}|$")
    ax_line.tick_params(labelsize=fig5_font)
    ax_line.xaxis.label.set_size(fig5_font)
    ax_line.yaxis.label.set_size(fig5_font)
    ax_line.yaxis.labelpad = 1
    ax_line.set_xlim(0, 30.5)
    ax_line.grid(alpha=0.16)
    ax_line.text(-0.075, 1.04, "(b)", transform=ax_line.transAxes, fontsize=fig5_font, fontweight="bold", clip_on=False)
    ax_line.legend(loc="upper left", fontsize=fig5_font, frameon=True)

    fig.subplots_adjust(left=0.08, right=0.925, top=0.965, bottom=0.08)
    save_figure(fig, "fig5_chess_align_mechanism", aliases=("fig4_chess_align_mechanism", "figure/fig4_align"))
    return error_df, trial_df, improvement_df


def spearman_alignment(improvement_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel, sub in improvement_df.groupby("channel"):
        clean = sub.dropna(subset=["improvement", "align_count", "empirical_abs_delta"])
        for x_col in ["align_count", "empirical_abs_delta"]:
            x = clean[x_col].to_numpy(dtype=float)
            y = clean["improvement"].to_numpy(dtype=float)
            spearman_rho, spearman_p = stats.spearmanr(x, y)
            pearson_r, pearson_p = stats.pearsonr(x, y)
            rows.append({
                "channel": channel,
                "x": x_col,
                "y": "aggregate_prediction_improvement",
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_rho": float(spearman_rho),
                "spearman_p": float(spearman_p),
                "n_trials": int(clean.shape[0]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "fig5_chess_align_correlations.csv", index=False)
    out.to_csv(OUT_DIR / "fig5_chess_align_spearman.csv", index=False)
    return out


def regression_alignment(improvement_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel, sub in improvement_df.groupby("channel"):
        clean = sub.dropna(subset=["improvement", "align_count", "empirical_abs_delta"]).copy()
        clean["interaction"] = clean["empirical_abs_delta"] * clean["align_count"]
        y = clean["improvement"].to_numpy(dtype=float)
        x = clean[["empirical_abs_delta", "align_count", "interaction"]].to_numpy(dtype=float)
        x_design = np.column_stack([np.ones(clean.shape[0]), x])
        beta, _, _, _ = np.linalg.lstsq(x_design, y, rcond=None)
        residual = y - x_design @ beta
        dof = max(1, x_design.shape[0] - x_design.shape[1])
        sigma2 = float((residual @ residual) / dof)
        cov = sigma2 * np.linalg.pinv(x_design.T @ x_design)
        se = np.sqrt(np.diag(cov))
        t_stat = beta / se
        p_vals = 2.0 * stats.t.sf(np.abs(t_stat), dof)
        for term, coef, term_se, t_value, p_value in zip(
            ["intercept", "empirical_abs_delta", "align_count", "interaction"],
            beta,
            se,
            t_stat,
            p_vals,
        ):
            rows.append({
                "channel": channel,
                "term": term,
                "coef": float(coef),
                "std_err": float(term_se),
                "t": float(t_value),
                "p_value": float(p_value),
                "n_trials": int(clean.shape[0]),
                "df_resid": int(dof),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "fig5_chess_align_regression.csv", index=False)
    return out


def regression_alignment_cases(case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel, sub in case_df.groupby("channel"):
        clean = sub.dropna(subset=["improvement", "align_count", "empirical_abs_delta"]).copy()
        clean["interaction"] = clean["empirical_abs_delta"] * clean["align_count"]
        y = clean["improvement"].to_numpy(dtype=float)
        x = clean[["empirical_abs_delta", "align_count", "interaction"]].to_numpy(dtype=float)
        x_design = np.column_stack([np.ones(clean.shape[0]), x])
        beta, _, _, _ = np.linalg.lstsq(x_design, y, rcond=None)
        residual = y - x_design @ beta
        dof = max(1, x_design.shape[0] - x_design.shape[1])
        sigma2 = float((residual @ residual) / dof)
        cov = sigma2 * np.linalg.pinv(x_design.T @ x_design)
        se = np.sqrt(np.diag(cov))
        t_stat = beta / se
        p_vals = 2.0 * stats.t.sf(np.abs(t_stat), dof)
        for term, coef, term_se, t_value, p_value in zip(
            ["intercept", "empirical_abs_delta", "align_count", "interaction"],
            beta,
            se,
            t_stat,
            p_vals,
        ):
            rows.append({
                "channel": channel,
                "term": term,
                "coef": float(coef),
                "std_err": float(term_se),
                "t": float(t_value),
                "p_value": float(p_value),
                "n_cases": int(clean.shape[0]),
                "df_resid": int(dof),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "fig5_chess_align_case_regression.csv", index=False)
    return out


def condition_mean_and_se(values: np.ndarray, condition: np.ndarray, cond: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.where(condition == cond)[0]
    vals = values[idx, 1:]
    return np.nanmean(vals, axis=0), np.nanstd(vals, axis=0, ddof=1) / np.sqrt(vals.shape[0])


def condition_pred_mean(pred: np.ndarray, condition: np.ndarray, cond: int) -> np.ndarray:
    return np.nanmean(pred[np.where(condition == cond)[0], 1:], axis=0)


def figure6_chess_trajectory(data: dict[str, np.ndarray]) -> None:
    trials = np.arange(1, 31)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 1.85), sharey=True)
    panels = [
        (axes[0], "ai", "base_ai", "full_ai", "(a)", "Confidence in AI"),
        (axes[1], "self", "base_self", "full_self", "(b)", "Self-confidence"),
    ]
    for ax, target, base_key, full_key, label, ylabel in panels:
        obs_mean, obs_se = condition_mean_and_se(data[target], data["condition"], 1)
        base = condition_pred_mean(data[base_key], data["condition"], 1)
        full = condition_pred_mean(data[full_key], data["condition"], 1)
        ax.errorbar(trials, obs_mean, yerr=obs_se, fmt="o", color=COLORS["data"], ecolor="#b8b8b8", capsize=2, markersize=3.5, label="Experiment Data")
        ax.plot(trials, base, color=COLORS["baseline"], marker="s", markersize=2.5, linewidth=1.2, label="Baseline Model")
        ax.plot(trials, full, color=COLORS["full"], marker="^", markersize=2.7, linewidth=1.2, label="Reformulated Model")
        ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=6, fontweight="bold")
        ax.set_xlim(0, 30)
        ax.set_ylim(0.32, 0.78)
        ax.set_xlabel("Trial")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(6)
        ax.yaxis.label.set_size(6)
    axes[0].legend(loc="lower left", fontsize=6, frameon=True, handlelength=1.6, borderpad=0.3, labelspacing=0.25)
    fig.tight_layout(w_pad=1.1, pad=0.4)
    save_figure(fig, "fig6_chess_model_fit", aliases=("fig2_chess_model_fit", "figure/fig5_traj"))


def truss_params(path: Path, model: str, track: str, names: list[str]) -> np.ndarray:
    df = pd.read_csv(path)
    row = df[(df["Model"] == model) & (df["track"] == track)].iloc[0]
    return row[names].to_numpy(dtype=float)


def condition_summary(observed: np.ndarray, *predicted: np.ndarray, condition_array: np.ndarray, cond: int = 1):
    idx = np.where(condition_array == cond)[0]
    trials = np.arange(1, observed.shape[1])
    obs = observed[idx, 1:]
    obs_mean = np.nanmean(obs, axis=0)
    obs_se = np.nanstd(obs, axis=0, ddof=1) / np.sqrt(obs.shape[0])
    pred_means = [np.nanmean(pred[idx, 1:], axis=0) for pred in predicted]
    return trials, obs_mean, obs_se, pred_means


def plot_row_panel(ax, trials, obs_mean, obs_se, baseline, reformulated, label, ylabel):
    ax.errorbar(trials, obs_mean, yerr=obs_se, fmt="o", color="#9a9a9a", ecolor="#b7b7b7", elinewidth=1.0, capsize=2, markersize=3.5, label="Experiment Data", zorder=1)
    ax.plot(trials, baseline, color="#1f3a93", marker="s", markersize=2.8, linewidth=1.2, label="Baseline Model", zorder=3)
    ax.plot(trials, reformulated, color="#b22222", marker="^", markersize=3.0, linewidth=1.2, label="Reformulated Model", zorder=4)
    ax.text(-0.10, 1.08, label, transform=ax.transAxes, fontsize=6, fontweight="bold", va="bottom")
    ax.set_xlim(0, 30)
    ax.set_ylim(0.25, 0.85)
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.16, linewidth=0.6)
    ax.tick_params(labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#6f6f6f")
        spine.set_linewidth(0.9)


def figure_chess_truss_row(data: dict[str, np.ndarray]) -> None:
    sys.path.insert(0, str(TRUSS_DIR))
    truss_og = load_module("manu_truss_og", TRUSS_DIR / "truss_og_benchmark.py")
    truss_detailed = load_module("manu_truss_detailed", TRUSS_DIR / "truss_detailed_subset_fit.py")
    truss_components = load_module("manu_truss_components", TRUSS_DIR / "truss_detailed_component_table.py")
    mats, _ = truss_detailed.build_subset_matrices()

    og_names = truss_og.PARAMETER_NAMES
    param_path = TRUSS_DIR / "data_folder_output" / "detailed_component_params.csv"
    og_ai_p = truss_params(param_path, "ORIG-BASE", "aiconf", og_names)
    og_self_p = truss_params(param_path, "ORIG-BASE", "selfconf", og_names)
    base_ai = truss_og.predict_og(og_ai_p, mats["ai_conf_matrix"], mats["baseline_e_tensor"])
    base_self = truss_og.predict_og(og_self_p, mats["self_conf_matrix"], mats["baseline_e_tensor"])

    variant = next(v for v in truss_components.VARIANTS if v.model == "PHYS-FULL")
    names = truss_components.param_names(variant)
    full_ai_p = truss_params(param_path, "PHYS-FULL", "aiconf", names)
    full_self_p = truss_params(param_path, "PHYS-FULL", "selfconf", names)
    full_ai = truss_components.predict_variant(
        full_ai_p,
        names,
        variant,
        mats["ai_conf_matrix"],
        mats["full_e_tensor"],
        mats["align_matrix"],
        False,
    )
    full_self = truss_components.predict_variant(
        full_self_p,
        names,
        variant,
        mats["self_conf_matrix"],
        mats["full_e_tensor"],
        mats["align_matrix"],
        True,
    )

    panels = [
        (condition_summary(data["ai"], data["base_ai"], data["full_ai"], condition_array=data["condition"]), "(a)", "Confidence in AI"),
        (condition_summary(data["self"], data["base_self"], data["full_self"], condition_array=data["condition"]), "(b)", "Self-confidence"),
        (condition_summary(mats["ai_conf_matrix"], base_ai, full_ai, condition_array=mats["condition_array"]), "(c)", "Confidence in AI"),
        (condition_summary(mats["self_conf_matrix"], base_self, full_self, condition_array=mats["condition_array"]), "(d)", "Self-confidence"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 3.0), sharex=True, sharey=False)
    for ax, (summary, label, ylabel) in zip(axes, panels):
        trials, obs_mean, obs_se, pred_means = summary
        plot_row_panel(ax, trials, obs_mean, obs_se, pred_means[0], pred_means[1], label, ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower left", fontsize=6, frameon=True, framealpha=0.9)
    fig.tight_layout(w_pad=1.0, pad=0.35)
    save_figure(fig, "fig_chess_truss_model_fit_row")


def write_checks(
    d_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    align_error: pd.DataFrame,
    align_trial: pd.DataFrame,
    corr_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    case_regression_df: pd.DataFrame,
) -> None:
    rows = []
    rows.append({"check": "Fig3 d sweep uses only modify trials", "value": int(d_df["n_modify_points"].iloc[0])})
    rows.append({"check": "Fig3 AI best d", "value": float(d_df.loc[d_df["ai_mse"].idxmin(), "d"])})
    rows.append({"check": "Fig3 self best d", "value": float(d_df.loc[d_df["self_mse"].idxmin(), "d"])})
    for model in ["Baseline Model", "Reformulated Model"]:
        for channel in ["AI channel", "Self channel"]:
            sub = align_error[(align_error["model"] == model) & (align_error["channel"] == channel)]
            val = weighted_mean(sub["abs_error"].to_numpy(dtype=float), sub["align_count"].to_numpy(dtype=float))
            rows.append({"check": f"Fig5 mean align abs error {model} {channel}", "value": float(val)})
    rows.append({"check": "Fig5 total alignment cases", "value": int(align_trial[align_trial["model"] == "Experiment Data"]["align_count"].sum())})
    rows.append({"check": "Fig4 modify experimental d(a)", "value": float(gap_df[(gap_df["model"] == "Experiment Data") & (gap_df["action"] == "modify")]["gap"].iloc[0])})
    for rec in corr_df.to_dict("records"):
        if rec["x"] == "align_count":
            rows.append({"check": f"Fig5 Spearman improvement vs align_count {rec['channel']}", "value": rec["spearman_rho"]})
            rows.append({"check": f"Fig5 Pearson improvement vs align_count {rec['channel']}", "value": rec["pearson_r"]})
    for rec in regression_df.to_dict("records"):
        if rec["term"] == "interaction":
            rows.append({"check": f"Fig5 regression interaction {rec['channel']}", "value": rec["coef"]})
    for rec in case_regression_df.to_dict("records"):
        if rec["term"] == "interaction":
            rows.append({"check": f"Fig5 case regression interaction {rec['channel']}", "value": rec["coef"]})
            rows.append({"check": f"Fig5 case regression interaction p {rec['channel']}", "value": rec["p_value"]})
    pd.DataFrame(rows).to_csv(OUT_DIR / "manu_figure_checks.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript figures from my_paper definitions.")
    parser.add_argument("--include-truss-row", action="store_true", help="Also regenerate the auxiliary chess+truss trajectory row.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_chess_predictions()
    d_df = figure3_d_sweep(data)
    gap_df = figure4_action_gap(data)
    align_error, align_trial, improvement_df = figure5_alignment(data)
    case_improvement_df = alignment_case_improvement(data)
    case_improvement_df.to_csv(OUT_DIR / "fig5_chess_align_case_improvement.csv", index=False)
    corr_df = spearman_alignment(improvement_df)
    regression_df = regression_alignment(improvement_df)
    case_regression_df = regression_alignment_cases(case_improvement_df)
    figure6_chess_trajectory(data)
    if args.include_truss_row:
        figure_chess_truss_row(data)
    write_checks(d_df, gap_df, align_error, align_trial, corr_df, regression_df, case_regression_df)


if __name__ == "__main__":
    main()

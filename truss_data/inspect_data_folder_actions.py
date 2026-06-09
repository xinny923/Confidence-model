from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "data_folder_output"


def matlab_cell(path: Path, key: str) -> np.ndarray:
    return loadmat(path, squeeze_me=True, struct_as_record=False)[key]


def as_list(value) -> list:
    return list(np.asarray(value, dtype=object).ravel())


def field(obj, name: str, default=0.0):
    return getattr(obj, name, default)


def topology_signature(mat: dict) -> Tuple[tuple, tuple]:
    nodes = []
    for node in as_list(mat["node_info"]):
        nodes.append((round(float(field(node, "x")), 4), round(float(field(node, "y")), 4)))

    members = []
    for member in as_list(mat["member_info"]):
        xs = np.asarray(field(member, "x"), dtype=float).ravel()
        ys = np.asarray(field(member, "y"), dtype=float).ravel()
        end_a = (round(float(xs[0]), 4), round(float(ys[0]), 4))
        end_b = (round(float(xs[-1]), 4), round(float(ys[-1]), 4))
        if end_b < end_a:
            end_a, end_b = end_b, end_a
        lw = round(float(field(member, "LW", 0.0)), 4)
        members.append((end_a, end_b, lw))

    return tuple(sorted(nodes)), tuple(sorted(members))


def load_topology(path: Path) -> tuple:
    return topology_signature(loadmat(path, squeeze_me=True, struct_as_record=False))


def participant_condition_and_row(pid: int) -> tuple[int, int]:
    if pid % 2 == 1:
        return 1, (pid + 1) // 2 - 1
    return 2, pid // 2 - 1


def participant_ids_with_csv() -> list[int]:
    ids = []
    for p in DATA_DIR.glob(".P*"):
        if not p.is_dir():
            continue
        m = re.fullmatch(r"\.P(\d+)", p.name)
        if m and (p / f"data{m.group(1)}.csv").exists():
            ids.append(int(m.group(1)))
    return sorted(ids)


def derive_suggestions(pids: Iterable[int], act_data: np.ndarray) -> dict[tuple[int, int], tuple]:
    suggestions: dict[tuple[int, int], tuple] = {}

    p77 = DATA_DIR / ".P77"
    for seq in range(4, 34):
        sugg_path = p77 / f"seq{seq}.sugg.mat"
        if sugg_path.exists():
            suggestions[(1, seq)] = load_topology(sugg_path)

    for pid in pids:
        cond, row = participant_condition_and_row(pid)
        for seq in range(4, 34):
            if (cond, seq) in suggestions:
                continue
            move2 = DATA_DIR / f".P{pid}" / f"seq{seq}.move2.mat"
            if not move2.exists() or row >= act_data[cond - 1].shape[0]:
                continue
            if int(act_data[cond - 1][row, seq - 4]) == 1:
                suggestions[(cond, seq)] = load_topology(move2)

    return suggestions


@dataclass
class TrialRecord:
    participant: int
    condition: int
    condition_row: int
    seq: int
    trial: int
    has_move1: bool
    has_move2: bool
    has_suggestion: bool
    old_choose_ai: float
    reconstructed_action: str
    align: float
    modified: float
    move2_equals_suggestion: float
    move2_equals_move1: float


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    act_data = matlab_cell(ROOT / "act_data.mat", "act_data")
    pids = participant_ids_with_csv()
    suggestions = derive_suggestions(pids, act_data)

    records = []
    for pid in pids:
        cond, row = participant_condition_and_row(pid)
        pdir = DATA_DIR / f".P{pid}"
        for seq in range(4, 34):
            move1 = pdir / f"seq{seq}.move1.mat"
            move2 = pdir / f"seq{seq}.move2.mat"
            has_move1 = move1.exists()
            has_move2 = move2.exists()
            has_suggestion = (cond, seq) in suggestions
            old = float("nan")
            if row < act_data[cond - 1].shape[0]:
                old = float(act_data[cond - 1][row, seq - 4])

            action = "missing"
            align = modified = move2_eq_sugg = move2_eq_move1 = float("nan")
            if has_move1 and has_move2 and has_suggestion:
                move1_sig = load_topology(move1)
                move2_sig = load_topology(move2)
                sugg_sig = suggestions[(cond, seq)]
                move2_eq_sugg = float(move2_sig == sugg_sig)
                move2_eq_move1 = float(move2_sig == move1_sig)
                align = float(move1_sig == sugg_sig)
                modified = float(not bool(move2_eq_sugg) and not bool(move2_eq_move1))
                if move2_eq_sugg:
                    action = "accept_ai"
                elif move2_eq_move1:
                    action = "reject_ai"
                else:
                    action = "modify"

            records.append(TrialRecord(
                participant=pid,
                condition=cond,
                condition_row=row,
                seq=seq,
                trial=seq - 3,
                has_move1=has_move1,
                has_move2=has_move2,
                has_suggestion=has_suggestion,
                old_choose_ai=old,
                reconstructed_action=action,
                align=align,
                modified=modified,
                move2_equals_suggestion=move2_eq_sugg,
                move2_equals_move1=move2_eq_move1,
            ))

    df = pd.DataFrame([r.__dict__ for r in records])
    usable = df[df["reconstructed_action"] != "missing"].copy()
    coverage = df.groupby("condition").agg(
        participants=("participant", "nunique"),
        trial_rows=("trial", "size"),
        usable_rows=("reconstructed_action", lambda s: int((s != "missing").sum())),
        suggestion_available=("has_suggestion", "sum"),
        move1_available=("has_move1", "sum"),
        move2_available=("has_move2", "sum"),
    ).reset_index()
    action_counts = usable.groupby(["condition", "reconstructed_action"]).size().reset_index(name="n")
    align_summary = usable.groupby("condition").agg(
        n=("trial", "size"),
        align_rate=("align", "mean"),
        modify_rate=("modified", "mean"),
        accept_ai_rate=("move2_equals_suggestion", "mean"),
        reject_ai_rate=("move2_equals_move1", "mean"),
    ).reset_index()
    old_match = usable.assign(
        reconstructed_choose_ai=(usable["reconstructed_action"] == "accept_ai").astype(float),
        old_match=lambda x: x["reconstructed_choose_ai"] == x["old_choose_ai"],
    ).groupby("condition").agg(
        n=("trial", "size"),
        old_act_match_rate=("old_match", "mean"),
    ).reset_index()

    df.to_csv(OUT_DIR / "data_folder_reconstructed_trials.csv", index=False)
    coverage.to_csv(OUT_DIR / "data_folder_coverage.csv", index=False)
    action_counts.to_csv(OUT_DIR / "data_folder_action_counts.csv", index=False)
    align_summary.to_csv(OUT_DIR / "data_folder_align_modify_summary.csv", index=False)
    old_match.to_csv(OUT_DIR / "data_folder_old_act_match.csv", index=False)

    print("Participant ids with csv:", pids)
    print("\nCoverage:")
    print(coverage.to_string(index=False))
    print("\nAction counts:")
    print(action_counts.to_string(index=False))
    print("\nAlign / modify summary:")
    print(align_summary.to_string(index=False))
    print("\nOld act_data match:")
    print(old_match.to_string(index=False))


if __name__ == "__main__":
    main()

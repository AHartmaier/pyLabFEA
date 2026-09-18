#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot stress-strain curves from calc_properties.py results."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MATERIAL = "ML-CPFEM-Random-Texture-cpfem"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_TRAIN_DATA = SCRIPT_DIR.parent / "Train_CPFEM" / "Data_Random_Texture.json"
DEFAULT_TRAIN_EPL_CRIT = 2.0e-3
SIG_NAMES = ["S11", "S22", "S33", "S23", "S13", "S12"]
EPL_NAMES = ["Ep11", "Ep22", "Ep33", "Ep23", "Ep13", "Ep12"]
UBC_NAMES = ["ux", "uy", "uz"]


def load_label(load: np.ndarray) -> str:
    direction = np.zeros_like(load)
    norm = np.linalg.norm(load)
    if norm > 0.0:
        direction = load / norm

    if np.allclose(direction, [1.0, 0.0, 0.0], atol=1.0e-3):
        return "uniaxial x"
    if np.allclose(direction, [0.0, 1.0, 0.0], atol=1.0e-3):
        return "uniaxial y"
    if np.allclose(direction, [0.0, 0.0, 1.0], atol=1.0e-3):
        return "uniaxial z"
    if np.count_nonzero(np.abs(direction) > 1.0e-3) == 2:
        signs = ["+" if value > 0.0 else "-" for value in direction[np.abs(direction) > 1.0e-3]]
        kind = "biaxial" if signs[0] == signs[1] else "shear-like"
        return f"{kind} [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]"
    return f"load [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]"


def grouped_load_indices(loads: np.ndarray, tolerance: float = 1.0e-10) -> list[tuple[str, np.ndarray]]:
    groups: list[tuple[str, np.ndarray]] = []
    used = np.zeros(len(loads), dtype=bool)
    for i, load in enumerate(loads):
        if used[i]:
            continue
        idx = np.nonzero(np.linalg.norm(loads - load, axis=1) <= tolerance)[0]
        used[idx] = True
        groups.append((load_label(load), idx))
    return groups


def split_load_groups(
    groups: list[tuple[str, np.ndarray]],
) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]]]:
    normal_groups = [(label, idx) for label, idx in groups if "shear-like" not in label]
    shear_groups = [(label, idx) for label, idx in groups if "shear-like" in label]
    return normal_groups, shear_groups


def equivalent_stress(stress: np.ndarray) -> np.ndarray:
    s11, s22, s33, s23, s13, s12 = stress.T
    return np.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
        + 3.0 * (s23**2 + s13**2 + s12**2)
    )


def equivalent_strain(strain: np.ndarray) -> np.ndarray:
    return np.sqrt(
        2.0
        * (
            np.sum(strain[:, 0:3] * strain[:, 0:3], axis=1)
            + 0.5 * np.sum(strain[:, 3:6] * strain[:, 3:6], axis=1)
        )
        / 3.0
    )


def short_case_name(name: str, max_length: int = 34) -> str:
    if len(name) <= max_length:
        return name
    return name[: max_length - 3] + "..."


def read_random_training_curves(
    train_data: Path,
    count: int,
    seed: int,
    peeq_offset: float,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    with train_data.open("r", encoding="utf-8") as fp:
        raw_data = json.load(fp)

    usable_curves = []
    for name, case_data in raw_data.items():
        results = case_data.get("Results", {})
        if not all(key in results for key in SIG_NAMES + EPL_NAMES):
            continue
        stress = np.array([results[key] for key in SIG_NAMES], dtype=np.float64).T
        plastic_strain = np.array([results[key] for key in EPL_NAMES], dtype=np.float64).T
        peeq = equivalent_strain(plastic_strain)
        seq = equivalent_stress(stress)
        valid = np.isfinite(peeq) & np.isfinite(seq)
        peeq = peeq[valid]
        seq = seq[valid]

        order = np.argsort(peeq)
        peeq = peeq[order]
        seq = seq[order]

        in_plastic_range = peeq >= peeq_offset
        if not np.any(in_plastic_range):
            continue

        shifted_peeq = peeq[in_plastic_range] - peeq_offset
        shifted_seq = seq[in_plastic_range]
        if peeq[0] <= peeq_offset <= peeq[-1]:
            seq_at_offset = np.interp(peeq_offset, peeq, seq)
            shifted_peeq = np.r_[0.0, shifted_peeq]
            shifted_seq = np.r_[seq_at_offset, shifted_seq]

        usable_curves.append((name, shifted_peeq, shifted_seq))

    if not usable_curves:
        raise ValueError(f"No usable shifted training load cases found in {train_data}")

    rng = random.Random(seed)
    return rng.sample(usable_curves, min(count, len(usable_curves)))


def plot_training_overlay(
    data: pd.DataFrame,
    groups: list[tuple[str, np.ndarray]],
    training_curves: list[tuple[str, np.ndarray, np.ndarray]],
    material: str,
    output_dir: Path,
    overlay_loads: str,
    peeq_offset: float,
) -> Path:
    peeq = data["PEEQ"].to_numpy(dtype=np.float64)
    mises = data["MISES"].to_numpy(dtype=np.float64)
    normal_groups, shear_groups = split_load_groups(groups)
    if overlay_loads == "normal":
        abaqus_groups = normal_groups
        load_text = "six normal/biaxial"
    elif overlay_loads == "shear":
        abaqus_groups = shear_groups
        load_text = "shear-like"
    else:
        abaqus_groups = groups
        load_text = "all"

    nonempty_abaqus_x = [
        group_x
        for _, idx in abaqus_groups
        for group_x in [peeq[idx][np.isfinite(peeq[idx])]]
        if len(group_x)
    ]
    if not nonempty_abaqus_x:
        raise ValueError("Selected Abaqus load cases have no finite PEEQ values.")
    abaqus_x_max = max(np.max(group_x) for group_x in nonempty_abaqus_x)

    nonempty_train_x = [train_peeq for _, train_peeq, _ in training_curves if len(train_peeq)]
    if not nonempty_train_x:
        raise ValueError("Selected training load cases have no finite shifted equivalent plastic strain values.")

    abaqus_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(abaqus_groups), 1)))
    train_colors = plt.cm.Dark2(np.linspace(0.0, 1.0, max(len(training_curves), 1)))

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for i, (label, idx) in enumerate(abaqus_groups):
        in_range = np.isfinite(peeq[idx]) & np.isfinite(mises[idx])
        if not np.any(in_range):
            continue
        ax.plot(
            peeq[idx][in_range],
            mises[idx][in_range],
            color=abaqus_colors[i],
            linewidth=1.4,
            alpha=0.85,
            label=f"Abaqus {label}",
        )

    for i, (name, train_peeq, train_seq) in enumerate(training_curves):
        in_range = np.isfinite(train_peeq) & np.isfinite(train_seq) & (train_peeq <= abaqus_x_max)
        if not np.any(in_range):
            continue
        ax.plot(
            train_peeq[in_range],
            train_seq[in_range],
            linestyle="--",
            linewidth=2.0,
            color=train_colors[i],
            label=f"train {short_case_name(name)}",
        )

    ax.set_xlabel("equiv. plastic strain, shifted by training epl_crit (.)")
    ax.set_ylabel("equiv. stress / Mises stress (MPa)")
    x_pad = 0.02 * abaqus_x_max
    ax.set_xlim(left=-x_pad, right=abaqus_x_max)
    ax.set_title(
        f"{material}: Abaqus {load_text} load cases vs random CPFEM training cases "
        f"(training PEEQ offset {peeq_offset:g})"
    )
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    overlay_path = output_dir / f"abq_{material}-training-overlay-{overlay_loads}-plastic-shifted.png"
    fig.savefig(overlay_path, dpi=200)
    return overlay_path


def plot_curves(
    data: pd.DataFrame,
    material: str,
    output_dir: Path,
    show: bool,
) -> list[tuple[str, np.ndarray]]:
    peeq = data["PEEQ"].to_numpy(dtype=np.float64)
    mises = data["MISES"].to_numpy(dtype=np.float64)
    loads = data[UBC_NAMES].to_numpy(dtype=np.float64)
    epl = data[EPL_NAMES].to_numpy(dtype=np.float64)

    groups = grouped_load_indices(loads)
    normal_groups, shear_groups = split_load_groups(groups)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for label, idx in normal_groups:
        ax.plot(peeq[idx], mises[idx], marker="o", markersize=3, linewidth=1.5, label=label)
    ax.set_xlabel("equiv. plastic strain (.)")
    ax.set_ylabel("Mises stress (MPa)")
    ax.set_title(material)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    normal_path = output_dir / f"abq_{material}-normal.png"
    fig.savefig(normal_path, dpi=200)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for label, idx in shear_groups:
        ax.plot(peeq[idx], mises[idx], marker="o", markersize=3, linewidth=1.5, label=label)
    ax.set_xlabel("equiv. plastic strain (.)")
    ax.set_ylabel("Mises stress (MPa)")
    ax.set_title(f"{material} shear-like load cases")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    shear_path = output_dir / f"abq_{material}-shear.png"
    fig.savefig(shear_path, dpi=200)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for label, idx in groups:
        plastic_norm = np.linalg.norm(epl[idx], axis=1)
        ax.plot(peeq[idx], plastic_norm, marker="o", markersize=3, linewidth=1.5, label=label)
    ax.set_xlabel("equiv. plastic strain (.)")
    ax.set_ylabel("plastic strain tensor norm (.)")
    ax.set_title(f"{material} plastic strain")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    epl_path = output_dir / f"abq_{material}-plastic-strain.png"
    fig.savefig(epl_path, dpi=200)

    print("Saved:", normal_path)
    print("Saved:", shear_path)
    print("Saved:", epl_path)
    if show:
        plt.show()
    else:
        plt.close("all")
    return groups


def find_result_file(results_dir: Path, material: str, result_file) -> Path:
    if result_file is not None:
        return result_file

    default_result = results_dir / f"abq_{material}-res.csv"
    if default_result.exists():
        return default_result

    matches = sorted(
        results_dir.glob(f"abq_{material}*-res.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Result file not found: {default_result}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CPFEM UMAT result curves.")
    parser.add_argument("material", nargs="?", default=DEFAULT_MATERIAL)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--result-file", type=Path, default=None)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--train-cases", type=int, default=3)
    parser.add_argument("--train-epl-crit", type=float, default=DEFAULT_TRAIN_EPL_CRIT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overlay-loads", choices=["normal", "shear", "all"], default="normal")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_file = find_result_file(args.results_dir, args.material, args.result_file)
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")

    data = pd.read_csv(result_file, header=0, sep=";").dropna()
    groups = plot_curves(data, args.material, args.results_dir, args.show)

    if args.train_cases > 0:
        if not args.train_data.exists():
            raise FileNotFoundError(f"Training JSON not found: {args.train_data}")
        training_curves = read_random_training_curves(
            args.train_data,
            args.train_cases,
            args.seed,
            args.train_epl_crit,
        )
        print(f"Selected {len(training_curves)} shifted training load case(s):")
        for name, _, _ in training_curves:
            print("  " + name)
        overlay_path = plot_training_overlay(
            data,
            groups,
            training_curves,
            args.material,
            args.results_dir,
            args.overlay_loads,
            args.train_epl_crit,
        )
        print("Saved:", overlay_path)
        if args.show:
            plt.show()
        else:
            plt.close("all")


if __name__ == "__main__":
    main()

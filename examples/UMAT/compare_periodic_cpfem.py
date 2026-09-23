#!/usr/bin/env python3
"""Compare a periodic ML UMAT simulation with a saved CPFEM reference.

Author: Ronak Shoghi
Date: 23 September 2026

Run from the repository root:
    python examples/UMAT/compare_periodic_cpfem.py

Replays the CPFEM stress path with NLGEOM=NO; the reference ODB is read-only.
Writes components_shifted.png (six-component curves) and cubes.png (paired
stress fields on a shared scale) to results/ beside this script by default.

Requires Abaqus/Standard, a compatible Fortran compiler, NumPy and Matplotlib.
Default inputs are relative to this script; use --help for input overrides.
Plastic shear strains must be engineering; total strain defaults to
Green-Lagrange with tensor shear, and stress defaults to Kirchhoff.
Working files are retained on failure or with --keep-work, otherwise removed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMPONENTS = ("11", "22", "33", "12", "13", "23")
DEFAULT_REFERENCE = HERE.parent / "Train_CPFEM/Data_Random_Texture_Test.json"
DEFAULT_MODEL = HERE / "models/abq_ML-CPFEM-Random-Texture-cpfem-svm.csv"
DEFAULT_UMAT = HERE / "ml_umat_cpfem_v2.f"
DEFAULT_CASE = "Us_A2B1C2D2E1F1_8b46e_5e411_Tx_Rnd"
DEFAULT_CPFEM_ODB = HERE / 'reference_data' / 'cpfem_test.odb'
DEFAULT_REFERENCE_LENGTH = .665


def strain_norm(values):
    """sqrt(2/3 eps:eps), with engineering shear entries (including trace)."""
    values = np.asarray(values)
    return np.sqrt(2.0 / 3.0 * (np.sum(values[..., :3] ** 2, axis=-1)
                              + 0.5 * np.sum(values[..., 3:] ** 2, axis=-1)))


def read_reference(dataset, case_key, case_index, limit, total_shear):
    with dataset.open() as stream:
        database = json.load(stream)
    keys = [key for key, case in database.items()
            if all(prefix + c in case.get("Results", {})
                   for prefix in ("S", "E", "Ep") for c in COMPONENTS)]
    if case_key is None:
        if not 0 <= case_index < len(keys):
            raise ValueError("Case index outside available range 0..%d" % (len(keys) - 1))
        case_key = keys[case_index]
    if case_key not in keys:
        raise ValueError("Case is missing six-component stress/strain/plastic histories")
    case = database[case_key]
    arrays = [np.array([case["Results"][prefix + c] for c in COMPONENTS], dtype=float).T
              for prefix in ("S", "E", "Ep")]
    if any(a.ndim != 2 or a.shape[1] != 6 for a in arrays):
        raise ValueError("Expected six scalar component histories")
    if any(a.shape != arrays[0].shape for a in arrays) or arrays[0].shape[0] < 2:
        raise ValueError("Reference arrays must have matching lengths >= 2")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Reference contains nonfinite values")
    stress, strain, plastic = arrays
    if total_shear == "tensor":
        strain[:, 3:] *= 2.0
    # The supplied test JSON's Ep shears are already engineering strain.
    norm = strain_norm(strain)
    crossing = np.flatnonzero(norm > limit)
    stop = int(crossing[0]) if len(crossing) else len(norm)
    if stop < 2:
        raise ValueError("Too few samples below the requested strain limit")
    stress, strain, plastic = [a[:stop].copy() for a in (stress, strain, plastic)]
    synthetic_origin = any(np.max(np.abs(a[0])) > 1.e-14 for a in (stress, strain, plastic))
    if synthetic_origin:
        stress, strain, plastic = [np.vstack((np.zeros((1, 6)), a))
                                  for a in (stress, strain, plastic)]
    time = np.linspace(0.0, 1.0, len(stress))
    metadata = case.get("Meta_Data") or {}
    return case_key, metadata, {
        "time": time.tolist(), "stress": stress.tolist(), "strain": strain.tolist(),
        "plastic_strain": plastic.tolist(), "source_samples_used": stop,
        "source_samples_total": int(len(norm)), "synthetic_zero_origin": synthetic_origin,
        "max_equivalent_total_strain": float(strain_norm(strain).max()),
    }


def elastic_matrix(props):
    """Export order is internal 11,22,33,23,13,12; return Abaqus order."""
    matrix = np.zeros((6, 6))
    matrix[:3, :3] = [[props[2], props[3], props[12]],
                     [props[3], props[10], props[13]],
                     [props[12], props[13], props[11]]]
    matrix[3, 3], matrix[4, 4], matrix[5, 5] = props[15], props[14], props[4]
    if props[10] < 0:
        matrix[:3, :3] = props[3]
        np.fill_diagonal(matrix[:3, :3], props[2])
        matrix[3:, 3:] = np.eye(3) * props[4]
    return matrix


def make_mesh(divisions, length):
    n = divisions
    label = lambda i, j, k: 1 + i + (n + 1) * j + (n + 1) ** 2 * k
    nodes, cells, pairs = [], [], []
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                number = label(i, j, k)
                nodes.append((number, length * i / n, length * j / n, length * k / n))
                if n in (i, j, k):
                    # Each positive-boundary node maps directly to one canonical
                    # independent node. No dependent DOF is reused at edges/corners.
                    base = (i % n, j % n, k % n)
                    pairs.append({"slave": number, "master": label(*base),
                                  "delta": [length * (v - b) / n
                                            for v, b in zip((i, j, k), base)]})
    for k in range(n):
        for j in range(n):
            for i in range(n):
                vertices = [(i, j, k), (i+1, j, k), (i+1, j+1, k), (i, j+1, k),
                            (i, j, k+1), (i+1, j, k+1), (i+1, j+1, k+1), (i, j+1, k+1)]
                cells.append((len(cells) + 1,) + tuple(label(*v) for v in vertices))
    controls = [{"label": len(nodes) + 1 + index, "component": comp}
                for index, comp in enumerate(COMPONENTS)]
    return nodes, cells, pairs, controls


def equation_terms(pair, dof, controls, length):
    terms = [(pair["slave"], dof + 1, 1.0), (pair["master"], dof + 1, -1.0)]
    # q = L * [eps11,eps22,eps33,gamma12,gamma13,gamma23].
    for axis, delta in enumerate(pair["delta"]):
        if delta:
            comp = "".join(str(v + 1) for v in sorted((dof, axis)))
            control = next(c["label"] for c in controls if c["component"] == comp)
            factor = 1.0 if dof == axis else 0.5
            terms.append((control, 1, -factor * delta / length))
    return terms


def input_deck(manifest, nodes, cells, props, constants_name):
    length = manifest["length_mm"]
    controls = manifest["controls"]
    ref = manifest["reference"]
    lines = ["*Heading", "** Periodic homogeneous ML cube: six-component stress replay",
             "** Small strain; reference time is normalized sample index, not seconds.",
             "*Preprint, echo=NO, model=NO, history=NO, contact=NO", "*Node"]
    lines += ["%d, %.16g, %.16g, %.16g" % node for node in nodes]
    lines += ["%d, %.16g, 0., 0." % (c["label"], length * (2 + i))
              for i, c in enumerate(controls)]
    lines += ["*Element, type=C3D8, elset=CUBE"]
    lines += [", ".join(str(v) for v in cell) for cell in cells]
    lines += ["*Nset, nset=CUBE_NODES, generate", "1, %d, 1" % len(nodes),
              "*Nset, nset=ANCHOR", "1", "*Nset, nset=MACRO"]
    lines += [", ".join(str(c["label"]) for c in controls)]
    for pair in manifest["periodic_pairs"]:
        for dof in range(3):
            terms = equation_terms(pair, dof, controls, length)
            lines += ["*Equation", str(len(terms))]
            for start in range(0, len(terms), 4):
                lines.append(", ".join("%d, %d, %.16g" % t for t in terms[start:start+4]))
    lines += ["*Orientation, name=GLOBAL_AXES", "1., 0., 0., 0., 1., 0.", "3, 0.",
              "*Solid Section, elset=CUBE, material=MATERIAL, orientation=GLOBAL_AXES", ",",
              "*Material, name=MATERIAL"]
    if manifest["material_mode"] == "ml":
        lines += ["*Depvar", "20", "*User Material, constants=%d, unsymm" % len(props),
                  "*Include, input=" + constants_name]
    else:
        c = elastic_matrix(props)
        # Abaqus TYPE=ORTHOTROPIC stiffness ordering D1111,D1122,D2222,
        # D1133,D2233,D3333,D1212,D1313,D2323.
        values = [c[0,0], c[0,1], c[1,1], c[0,2], c[1,2], c[2,2], c[3,3], c[4,4], c[5,5]]
        lines += ["*Elastic, type=ORTHOTROPIC", ", ".join("%.16g" % v for v in values[:8]),
                  "%.16g" % values[8]]
    for i, comp in enumerate(COMPONENTS):
        lines += ["*Amplitude, name=STRESS_" + comp + ", time=STEP TIME"]
        values = [(t, row[i]) for t, row in zip(ref["time"], ref["stress"])]
        for start in range(0, len(values), 4):
            lines.append(", ".join("%.16g, %.16g" % pair for pair in values[start:start+4]))
    lines += ["*Boundary", "ANCHOR, 1, 3, 0."]
    # Unused control DOFs are not activated by elements/equations.
    lines += ["*Time Points, name=REFERENCE"]
    times = ref["time"]
    for start in range(0, len(times), 8):
        lines.append(", ".join("%.16g" % v for v in times[start:start+8]))
    increment = min(0.01, 1.0 / (len(times) - 1))
    lines += ["*Step, name=STRESS_REPLAY, nlgeom=NO, inc=20000, unsymm=YES",
              "*Static", "%.16g, 1., 1.e-9, %.16g" % (increment, increment)]
    for control in controls:
        lines += ["*Cload, amplitude=STRESS_" + control["component"],
                  "%d, 1, %.16g" % (control["label"], length ** 2)]
    lines += ["*Restart, write, frequency=0",
              "*Output, field, time points=REFERENCE, time marks=YES",
              "*Node Output", "U, RF", "*Element Output, elset=CUBE",
              "S, E, IVOL" + (", SDV" if manifest["material_mode"] == "ml" else ""),
              "*Output, history, frequency=1", "*Node Output, nset=MACRO", "U1, CF1",
              "*End Step", ""]
    return "\n".join(lines)



def _init_plotting():
    """Delay Matplotlib import until the temporary cache directory is configured."""
    global plt, Normalize, Line2D, Rectangle, MaxNLocator, Poly3DCollection
    if 'plt' in globals():
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection


_HEX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
              (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def _six_columns(values, name, count=None):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 6 or len(values) == 0
            or (count is not None and len(values) != count)
            or not np.isfinite(values).all()):
        raise ValueError(name + " must be a finite, nonempty n-by-6 array")
    return values


def shifted_reference(reference, offset):
    """Return shifted total strain, keeping the recorded elastic strain.

    p = sqrt(2/3 * (Ep11^2 + Ep22^2 + Ep33^2
                     + (gamma_p12^2 + gamma_p13^2 + gamma_p23^2)/2)).
    Ep_shifted = Ep * max(0, 1 - offset/p), zero where p is zero.
    E_shifted = (E - Ep) + Ep_shifted.

    This is the fixed training convention, not a fitted curve translation.
    The original inputs are not modified.
    """
    if not np.isfinite(offset) or offset < 0.:
        raise ValueError("The training plastic-strain offset must be finite and nonnegative")
    strain = _six_columns(reference["strain"], "CPFEM total strain")
    plastic = _six_columns(reference["plastic_strain"], "CPFEM plastic strain", len(strain))
    equivalent = np.sqrt((2. / 3.) * np.sum(
        plastic ** 2 * np.array([1., 1., 1., .5, .5, .5]), axis=1))
    factor = np.zeros_like(equivalent)
    np.divide(offset, equivalent, out=factor, where=equivalent > 0.)
    factor = np.maximum(0., 1. - factor)
    return strain - plastic + plastic * factor[:, None]


def _history_columns(history, prefix):
    history = np.atleast_1d(history)
    names = [prefix + component for component in COMPONENTS]
    missing = set(names).difference(history.dtype.names or ())
    if missing:
        raise ValueError("Missing ML history columns: " + ", ".join(sorted(missing)))
    return _six_columns(np.column_stack([history[name] for name in names]), prefix)


def plot_components(reference, ml_history, offset, output_path, measures_note=None):
    """Save one PNG with six native-sample stress versus shifted-strain curves.

    reference: dict of time, stress, strain and plastic_strain arrays.
    ml_history: structured array containing mean_Sij and macro_Eij columns.
    The shift applies only to the CPFEM plastic part of total strain.
    """
    _init_plotting()
    strain = shifted_reference(reference, offset)
    stress = _six_columns(reference["stress"], "CPFEM stress", len(strain))
    ml_stress = _history_columns(ml_history, "mean_S")
    ml_strain = _history_columns(ml_history, "macro_E")
    if len(ml_stress) != len(ml_strain):
        raise ValueError("ML stress and strain sample counts differ")
    if measures_note is None:
        measures_note = ("Recorded CPFEM: Kirchhoff stress / Green–Lagrange strain; "
                         "ML UMAT: Cauchy stress / small strain.")
    style = {"font.family": "DejaVu Sans", "font.size": 11,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.linewidth": .8, "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(15.2, 10.))
        fig.subplots_adjust(left=.073, right=.984, bottom=.19, top=.83,
                            wspace=.30, hspace=.45)
        for index, ax in enumerate(axes.flat):
            component = COMPONENTS[index]
            ax.plot(100. * strain[:, index], stress[:, index],
                    color="black", linewidth=2.3, label="CPFEM", zorder=3)
            ax.plot(100. * ml_strain[:, index], ml_stress[:, index],
                    color="red", linewidth=2.0, linestyle=(0, (5, 3)),
                    label="ML UMAT", zorder=4)
            symbol = r"\varepsilon" if index < 3 else r"\gamma"
            kind = "Normal strain" if index < 3 else "Engineering shear strain"
            ax.set_xlabel(kind + " $" + symbol + "_{" + component + "}$ (%)", labelpad=8)
            ax.set_ylabel("Stress component " + component + " (MPa)", labelpad=8)
            ax.set_title("Component " + component, loc="left", fontsize=12, weight="bold", pad=12)
            ax.grid(True, color="#dfdfdf", linewidth=.65, zorder=0)
            ax.axhline(0., color="#b8b8b8", linewidth=.6, zorder=1)
            ax.axvline(0., color="#b8b8b8", linewidth=.6, zorder=1)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.tick_params(labelsize=10)
            ax.ticklabel_format(axis="both", style="plain", useOffset=False)
            ax.margins(x=.065, y=.08)
        fig.suptitle("CPFEM and ML UMAT: component stress–strain response",
                     x=.073, y=.968, ha="left", fontsize=19, weight="bold")
        fig.text(.073, .922,
                 "CPFEM plastic-strain shift = {:g} ({:g}%); elastic strain preserved".format(
                     offset, offset * 100.), fontsize=12, color="#454545")
        fig.legend(handles=[Line2D([], [], color="black", linewidth=2.3),
                            Line2D([], [], color="red", linewidth=2., linestyle=(0, (5, 3)))],
                   labels=["CPFEM", "ML UMAT"], loc="upper left",
                   bbox_to_anchor=(.067, .896), frameon=False, ncol=2,
                   fontsize=12, handlelength=3.2, columnspacing=2.5)
        fig.text(.073, .095,
                 "Fixed training shift applied to CPFEM only; native simulation samples shown.",
                 fontsize=10, color="#505050")
        fig.text(.073, .068, measures_note, fontsize=9.3, color="#505050")
        fig.text(.073, .040,
                 "The ML cube follows the CPFEM stress path; strain response is the predicted comparison.",
                 fontsize=9.3, color="#505050")
        fig.savefig(output_path, dpi=220, format="png")
        plt.close(fig)


def _cube_surface(snapshot):
    """Keep exterior C3D8 faces, matching coincident nodes across instances."""
    if tuple(snapshot["components"]) != COMPONENTS:
        raise ValueError("Snapshot components must be 11,22,33,12,13,23")
    nodes, elements = snapshot["nodes"], snapshot["elements"]
    index = {(node["instance"], node["label"]): i for i, node in enumerate(nodes)}
    if len(index) != len(nodes):
        raise ValueError("Duplicate instance/node labels")
    coordinates = np.asarray([node["coordinates"] for node in nodes], dtype=float)
    displacement = np.asarray([node["displacement"] for node in nodes], dtype=float)
    stress = _six_columns([element["mean_stress"] for element in elements], "Element stress")
    if (coordinates.shape != (len(nodes), 3) or displacement.shape != coordinates.shape
            or not np.isfinite(coordinates).all() or not np.isfinite(displacement).all()):
        raise ValueError("Snapshot coordinates and displacements must be finite n-by-3 arrays")
    length = float(np.max(np.ptp(coordinates, axis=0)))
    if length <= 0.:
        raise ValueError("Snapshot has no spatial extent")
    # Origin subtraction also avoids quantization overflow for remote origins.
    point_keys = [tuple(row) for row in np.rint(
        (coordinates - coordinates.min(axis=0)) / (length * 1.e-9)).astype(np.int64)]
    faces_by_location = {}
    for owner, element in enumerate(elements):
        if not element["type"].startswith("C3D8") or len(element["connectivity"]) != 8:
            raise ValueError("Cube plotting supports eight-node C3D8-family solids")
        connectivity = [index[(element["instance"], label)] for label in element["connectivity"]]
        for local_face in _HEX_FACES:
            face = tuple(connectivity[i] for i in local_face)
            key = tuple(sorted(point_keys[i] for i in face))
            faces_by_location.setdefault(key, []).append((face, owner))
    if any(len(entries) > 2 for entries in faces_by_location.values()):
        raise ValueError("More than two elements share a face")
    surface = [entries[0] for entries in faces_by_location.values() if len(entries) == 1]
    if not surface:
        raise ValueError("No exterior faces found")
    volumes = np.asarray([element.get("volume") for element in elements], dtype=float)
    if not np.isfinite(volumes).all() or np.any(volumes <= 0.):
        raise ValueError("Volume-mean cube labels require positive physical element volumes")
    mean = np.average(stress, axis=0, weights=volumes)
    recorded_mean = np.asarray(snapshot["mean_stress"], dtype=float)
    if recorded_mean.shape != (6,) or not np.allclose(mean, recorded_mean, atol=1.e-7, rtol=1.e-7):
        raise ValueError("Snapshot volume mean disagrees with its element stresses and volumes")
    return {"coordinates": coordinates, "displacement": displacement,
            "stress": stress, "mean_stress": mean, "length": length,
            "faces": np.asarray([face for face, _ in surface], dtype=int),
            "owners": np.asarray([owner for _, owner in surface], dtype=int)}


def plot_cubes(cpfem_snapshot, ml_snapshot, output_path, deformation_scale=1., color_limit=None):
    """Save one 12-cube PNG, using one absolute MPa color scale everywhere.

    Both inputs are dictionaries from the actual Abaqus snapshot extractor.
    All element stress components in both volumes define the shared symmetric
    range, including interior elements. No percentile clipping is performed.
    Each cube also shows its physical volume mean as a colored swatch.
    """
    _init_plotting()
    if not np.isfinite(deformation_scale) or deformation_scale <= 0.:
        raise ValueError("Deformation scale must be finite and positive")
    surfaces = [_cube_surface(snapshot) for snapshot in (cpfem_snapshot, ml_snapshot)]
    bounds = [(surface["coordinates"].min(axis=0), surface["coordinates"].max(axis=0))
              for surface in surfaces]
    if not np.allclose(bounds[0], bounds[1], atol=1.e-6, rtol=1.e-6):
        raise ValueError("The cubes must share physical reference extents and origin")
    maximum = max(float(np.max(np.abs(surface["stress"]))) for surface in surfaces)
    if color_limit is None:
        if maximum == 0.:
            color_limit = 1.
        else:
            rounding = 10. ** np.floor(np.log10(maximum)) / 5.
            color_limit = np.ceil(maximum / rounding) * rounding
    if not np.isfinite(color_limit) or color_limit <= 0. or color_limit < maximum:
        raise ValueError("Color limit must be finite, positive and include every stress value ({:g} MPa)".format(maximum))
    normalization = Normalize(vmin=-color_limit, vmax=color_limit)
    cmap = plt.get_cmap("viridis")
    for surface in surfaces:
        surface["deformed"] = surface["coordinates"] + deformation_scale * surface["displacement"]
    points = np.concatenate([surface["deformed"] for surface in surfaces])
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2.
    width = float(np.max(upper - lower)) * 1.04
    lower, upper = center - width / 2., center + width / 2.
    style = {"font.family": "DejaVu Sans", "font.size": 11, "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(17., 13.1))
        grid = fig.add_gridspec(3, 4, left=.033, right=.974, bottom=.165,
                               top=.858, wspace=.025, hspace=.22)
        for component in range(6):
            row, pair = divmod(component, 2)
            for source, (surface, label, identity_color) in enumerate(
                    zip(surfaces, ("CPFEM", "ML UMAT"), ("black", "red"))):
                ax = fig.add_subplot(grid[row, pair * 2 + source], projection="3d")
                stress = surface["stress"][surface["owners"], component]
                polygons = Poly3DCollection(surface["deformed"][surface["faces"]],
                                           facecolors=cmap(normalization(stress)),
                                           edgecolors="#555555", linewidths=.18,
                                           antialiased=True, zsort="average")
                ax.add_collection3d(polygons)
                ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                       zlim=(lower[2], upper[2]))
                ax.set_box_aspect((1, 1, 1), zoom=1.23)
                ax.view_init(elev=23, azim=-57)
                ax.set_proj_type("ortho")
                ax.set_axis_off()
                ax.set_title(label + "  ·  S" + COMPONENTS[component],
                             color=identity_color, fontsize=12, pad=0., weight="bold")
                mean = surface["mean_stress"][component]
                position = ax.get_position()
                # Figure-level swatches avoid 3D patch transformation/shading.
                x = position.x0 + position.width * .13
                y = position.y0 - .010
                fig.add_artist(Rectangle((x, y), .0135, .0115,
                                         transform=fig.transFigure,
                                         facecolor=cmap(normalization(mean)),
                                         edgecolor="#555555", linewidth=.6, clip_on=False))
                fig.text(x + .020, y + .001,
                         r"$\langle S_{" + COMPONENTS[component] + r"}\rangle_V$ = "
                         + "{:+.2f} MPa".format(mean), fontsize=10, color="#333333")
        fig.suptitle("CPFEM microstructure and its homogenized ML representation",
                     x=.045, y=.971, ha="left", fontsize=20, weight="bold")
        fig.text(.045, .935,
                 "Six Cauchy stress components at the matched loading state", fontsize=13, color="#444444")
        fig.text(.045, .906,
                 "One color scale for all 12 cubes and all mean-stress swatches: the same stress has the same color.",
                 fontsize=11, color="#444444")
        bar_ax = fig.add_axes([.23, .088, .54, .018])
        scalar = plt.cm.ScalarMappable(norm=normalization, cmap=cmap)
        scalar.set_array([])
        colorbar = fig.colorbar(scalar, cax=bar_ax, orientation="horizontal",
                               ticks=np.linspace(-color_limit, color_limit, 9))
        colorbar.set_label("Element-mean Cauchy stress component (MPa)", labelpad=8, fontsize=11)
        colorbar.ax.tick_params(labelsize=10)
        fig.text(.045, .027,
                 "CPFEM: grain-resolved field    |    ML UMAT: homogenized material    |    "
                 "Actual displacement ×{:g}; common size and camera".format(deformation_scale),
                 fontsize=10, color="#505050")
        fig.savefig(output_path, dpi=220, format="png")
        plt.close(fig)
    return {"vmin": -float(color_limit), "vmax": float(color_limit),
            "colormap": "viridis", "mean_stresses": [surface["mean_stress"].tolist() for surface in surfaces]}


# Abaqus uses its own Python interpreter; this source is written only to the run's temporary directory.
ODB_WORKER_SOURCE = r'''"""Temporary Abaqus ODB worker; compatible with Abaqus Python 2.7 and 3.x."""
from __future__ import print_function

import bisect
import json
import math
import os
import sys

COMPONENTS = ('11', '22', '33', '12', '13', '23')


def _data(value):
    try:
        raw = value.data
    except Exception:
        raw = value.dataDouble
    result = ([float(raw)] if isinstance(raw, (float, int)) else
              [float(item) for item in raw])
    if not all(not math.isnan(item) and not math.isinf(item) for item in result):
        raise ValueError('Non-finite ODB field value.')
    return result


def _owner(value):
    return getattr(getattr(value, 'instance', None), 'name', '')


def _ip_key(value):
    return (_owner(value), int(value.elementLabel),
            int(getattr(value, 'integrationPoint', 0)),
            int(getattr(getattr(value, 'sectionPoint', None), 'number', 0)))


def _mesh(odb):
    nodes, elements = {}, {}
    for instance in odb.rootAssembly.instances.values():
        connected = set()
        for element in instance.elements:
            if not element.type.startswith('C3D8'):
                continue
            key = (instance.name, int(element.label))
            connectivity = [(instance.name, int(label)) for label in element.connectivity]
            if len(connectivity) != 8:
                raise ValueError('Only eight-node C3D8-family cubes are supported.')
            elements[key] = (element.type, connectivity)
            connected.update(connectivity)
        for node in instance.nodes:
            key = (instance.name, int(node.label))
            if key in connected:
                nodes[key] = [float(x) for x in node.coordinates]
    if not elements:
        raise ValueError('No C3D8-family solid elements in ODB.')
    lower = [min(x[i] for x in nodes.values()) for i in range(3)]
    upper = [max(x[i] for x in nodes.values()) for i in range(3)]
    lengths = [upper[i] - lower[i] for i in range(3)]
    if min(lengths) <= 0.0:
        raise ValueError('Degenerate three-dimensional cube mesh.')
    tol = max(lengths) * 1.e-6
    corners = dict((key, x) for key, x in nodes.items()
                   if all(min(abs(x[i] - lower[i]), abs(x[i] - upper[i])) < tol
                          for i in range(3)))
    if len(corners) != 8:
        raise ValueError('Expected eight bounding-box corners, found ' + str(len(corners)))
    return {'nodes': nodes, 'elements': elements, 'corners': corners,
            'lower': lower, 'upper': upper}


def _nodal(frame, name):
    if name not in frame.fieldOutputs:
        raise ValueError('Missing nodal output: ' + name)
    values = {}
    for value in frame.fieldOutputs[name].values:
        key = (_owner(value), int(value.nodeLabel))
        if key in values:
            raise ValueError('Duplicate nodal field key: ' + str(key))
        values[key] = _data(value)
    return values


def _kinematics(frame, mesh):
    corners = mesh['corners']
    displacement = dict(((_owner(v), int(v.nodeLabel)), _data(v))
                        for v in frame.fieldOutputs['U'].values
                        if (_owner(v), int(v.nodeLabel)) in corners)
    if set(displacement) != set(corners):
        raise ValueError('Missing corner displacement.')
    center = [(mesh['lower'][i] + mesh['upper'][i]) * 0.5 for i in range(3)]
    mean_u = [sum(u[i] for u in displacement.values()) / 8.0 for i in range(3)]
    grad = [[sum((displacement[key][i] - mean_u[i]) * (corners[key][j] - center[j])
                 for key in corners) / sum((x[j] - center[j]) ** 2 for x in corners.values())
             for j in range(3)] for i in range(3)]
    residual = max(abs(displacement[key][i] - mean_u[i] -
                       sum(grad[i][j] * (corners[key][j] - center[j]) for j in range(3)))
                   for key in corners for i in range(3))
    F = [[grad[i][j] + (1.0 if i == j else 0.0) for j in range(3)] for i in range(3)]
    green = [[0.5 * (sum(F[k][i] * F[k][j] for k in range(3)) -
                     (1.0 if i == j else 0.0)) for j in range(3)] for i in range(3)]
    J = (F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1]) -
         F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0]) +
         F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]))
    if J <= 0.0:
        raise ValueError('Nonpositive macroscopic deformation Jacobian.')
    return {'macro_strain': [grad[0][0], grad[1][1], grad[2][2],
                             grad[0][1] + grad[1][0], grad[0][2] + grad[2][0],
                             grad[1][2] + grad[2][1]],
            'macro_green_strain': [green[0][0], green[1][1], green[2][2],
                                   2.0 * green[0][1], 2.0 * green[0][2], 2.0 * green[1][2]],
            'macro_deformation_gradient': F, 'macro_jacobian': J,
            'corner_affine_fit_residual_mm': residual}


def _field(frame, name, mesh, tensor=False):
    from abaqusConstants import INTEGRATION_POINT
    if name not in frame.fieldOutputs:
        raise ValueError('Missing integration-point field: ' + name)
    field = frame.fieldOutputs[name]
    indices = None
    if tensor:
        labels = list(field.componentLabels)
        indices = [labels.index(name + item) for item in COMPONENTS]
    result = {}
    for value in field.getSubset(position=INTEGRATION_POINT).values:
        key = _ip_key(value)
        if key[:2] not in mesh['elements']:
            continue
        if key in result:
            raise ValueError('Duplicate integration-point key in ' + name)
        if tensor:
            try:
                local = value.localCoordSystem
            except Exception:
                local = value.localCoordSystemDouble
            if local is not None and any(abs(float(local[i][j]) - (1.0 if i == j else 0.0)) > 1.e-7
                                         for i in range(3) for j in range(3)):
                raise ValueError('Local-coordinate ' + name +
                                 ' detected; global tensor output is required.')
        raw = _data(value)
        result[key] = [raw[i] for i in indices] if indices is not None else raw
    if not result:
        raise ValueError('Empty integration-point field: ' + name)
    return result


def _geometric_weights(frame, mesh, stress_keys):
    """Current integration-point volumes for fully integrated C3D8 elements."""
    import numpy as np
    keys = sorted(mesh['elements'])
    if any(mesh['elements'][key][0] != 'C3D8' for key in keys):
        raise ValueError('Without IVOL, current-volume integration requires full C3D8 elements.')
    displacement = _nodal(frame, 'U')
    current = np.asarray([[np.asarray(mesh['nodes'][node]) + np.asarray(displacement[node])
                           for node in mesh['elements'][key][1]] for key in keys])
    signs = np.asarray([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    gauss = signs / math.sqrt(3.0)
    derivatives = np.empty((8, 8, 3))
    for ip in range(8):
        for node in range(8):
            for axis in range(3):
                other = [i for i in range(3) if i != axis]
                derivatives[ip, node, axis] = (0.125 * signs[node, axis] *
                    (1.0 + signs[node, other[0]] * gauss[ip, other[0]]) *
                    (1.0 + signs[node, other[1]] * gauss[ip, other[1]]))
    a = np.einsum('enk,pnj->epkj', current, derivatives)
    determinants = (a[:, :, 0, 0] * (a[:, :, 1, 1] * a[:, :, 2, 2] - a[:, :, 1, 2] * a[:, :, 2, 1]) -
                    a[:, :, 0, 1] * (a[:, :, 1, 0] * a[:, :, 2, 2] - a[:, :, 1, 2] * a[:, :, 2, 0]) +
                    a[:, :, 0, 2] * (a[:, :, 1, 0] * a[:, :, 2, 1] - a[:, :, 1, 1] * a[:, :, 2, 0]))
    if not np.isfinite(determinants).all() or np.any(determinants <= 0.0):
        raise ValueError('Invalid current C3D8 Jacobian determinant.')
    index = dict((key, i) for i, key in enumerate(keys))
    if any(not 1 <= key[2] <= 8 for key in stress_keys):
        raise ValueError('Unexpected C3D8 integration-point number.')
    return dict((key, float(determinants[index[key[:2]], key[2] - 1])) for key in stress_keys)


def _fields(frame, mesh):
    stress = _field(frame, 'S', mesh, tensor=True)
    strain_name = next((name for name in ('E', 'LE') if name in frame.fieldOutputs), None)
    strain = _field(frame, strain_name, mesh, tensor=True) if strain_name else None
    if strain is not None and set(strain) != set(stress):
        raise ValueError('Stress and strain integration points differ.')
    if 'IVOL' in frame.fieldOutputs:
        weight = dict((key, value[0]) for key, value in _field(frame, 'IVOL', mesh).items())
        weighting = 'IVOL'
    else:
        weight = _geometric_weights(frame, mesh, stress)
        weighting = 'current C3D8 Gauss-point Jacobian volumes from X+U'
    if set(weight) != set(stress) or not all(v > 0.0 for v in weight.values()):
        raise ValueError('Stress/volume integration keys differ or volume is nonpositive.')
    total = sum(weight.values())
    mean_s = [sum(weight[key] * stress[key][i] for key in stress) / total for i in range(6)]
    mean_e = ([sum(weight[key] * strain[key][i] for key in strain) / total for i in range(6)]
              if strain is not None else None)
    return {'stress': stress, 'strain': strain, 'weights': weight,
            'mean_stress': mean_s, 'mean_strain': mean_e, 'total_volume': total,
            'weighting': weighting, 'strain_field': strain_name}


def _snapshot(odb, source, step_name, frame_index, mesh, fields=None):
    frame = odb.steps[step_name].frames[frame_index]
    data = fields if fields is not None else _fields(frame, mesh)
    displacement = _nodal(frame, 'U')
    groups = {}
    for key in data['stress']:
        groups.setdefault(key[:2], []).append(key)
    elements = []
    for key in sorted(mesh['elements']):
        kind, connectivity = mesh['elements'][key]
        ips = groups.get(key, [])
        if not ips:
            raise ValueError('Missing solid-element stress: ' + str(key))
        volume = sum(data['weights'][ip] for ip in ips)
        mean_s = [sum(data['weights'][ip] * data['stress'][ip][i] for ip in ips) / volume
                  for i in range(6)]
        mean_e = ([sum(data['weights'][ip] * data['strain'][ip][i] for ip in ips) / volume
                   for i in range(6)] if data['strain'] is not None else None)
        elements.append({'instance': key[0], 'label': key[1], 'type': kind,
                         'connectivity': [node[1] for node in connectivity],
                         'mean_stress': mean_s, 'mean_strain': mean_e, 'volume': volume})
    nodes = [{'instance': key[0], 'label': key[1], 'coordinates': mesh['nodes'][key],
              'displacement': displacement[key]} for key in sorted(mesh['nodes'])]
    result = {'source_odb': os.path.abspath(source), 'opened_odb': os.path.abspath(odb.path),
              'step': step_name, 'frame_index': frame_index, 'frame_time': float(frame.frameValue),
              'components': list(COMPONENTS), 'nodes': nodes, 'elements': elements,
              'node_count': len(nodes), 'element_count': len(elements),
              'mean_stress': data['mean_stress'], 'mean_strain': data['mean_strain'],
              'total_volume': data['total_volume'], 'strain_field': data['strain_field'],
              'strain_convention': 'engineering shear', 'stress_unit': 'MPa',
              'length_unit': 'mm', 'weighting': data['weighting']}
    result.update(_kinematics(frame, mesh))
    return result


def _open_odb(source, work_dir, name):
    import odbAccess
    source = os.path.abspath(source)
    # Abaqus 2022's C++ wrappers reject json.load's Python-2 unicode paths.
    if sys.version_info[0] < 3:
        source = source.encode('utf-8')
        work_dir = work_dir.encode('utf-8')
    if not os.path.isfile(source):
        raise ValueError('ODB does not exist: ' + source)
    opened = source
    if odbAccess.isUpgradeRequiredForOdb(upgradeRequiredOdbPath=source):
        opened = os.path.join(work_dir, name + '_upgraded.odb')
        print('Upgrading a temporary copy of ' + source)
        sys.stdout.flush()
        odbAccess.upgradeOdb(existingOdbPath=source, upgradedOdbPath=opened)
    return odbAccess.openOdb(path=opened, readOnly=True)


def _target(reference, time):
    times, stress = reference['time'], reference['stress']
    if time <= times[0]:
        return list(stress[0])
    if time >= times[-1]:
        return list(stress[-1])
    upper = bisect.bisect_right(times, time)
    lower = upper - 1
    fraction = (time - times[lower]) / float(times[upper] - times[lower])
    return [stress[lower][i] + fraction * (stress[upper][i] - stress[lower][i])
            for i in range(6)]


def _ml_rows(odb, manifest, mesh, source):
    step_name = list(odb.steps.keys())[-1]
    frames = odb.steps[step_name].frames
    expected = float(manifest.get('expected_completion_time', 1.0))
    if not frames or abs(float(frames[-1].frameValue) - expected) > 1.e-8:
        raise ValueError('ML ODB does not reach the complete prescribed loading interval.')
    sta_path = os.path.splitext(source)[0] + '.sta'
    if not os.path.isfile(sta_path):
        raise ValueError('ML completion status is missing: ' + sta_path)
    with open(sta_path) as handle:
        if 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' not in handle.read().upper():
            raise ValueError('ML .sta file does not confirm successful analysis completion.')
    length = float(manifest['length_mm'])
    labels = dict((item['component'], int(item['label'])) for item in manifest['controls'])
    rows = []
    max_error, max_ratio, max_periodic = 0.0, 0.0, 0.0
    last_data = None
    for frame in frames:
        time = float(frame.frameValue)
        if 'S' not in frame.fieldOutputs:
            if abs(time) > 1.e-12 or any(abs(x) > 1.e-12 for x in _target(manifest['reference'], time)):
                raise ValueError('Nonzero ML frame is missing stress output.')
            rows.append({'pseudo_time': time, 'mean_stress': [0.0] * 6,
                         'macro_strain': [0.0] * 6, 'mean_strain': [0.0] * 6,
                         'target_stress': [0.0] * 6, 'periodic_residual_mm': 0.0})
            continue
        displacement_by_key = _nodal(frame, 'U')
        displacements = {}
        for key, value in displacement_by_key.items():
            if manifest.get('instance_name') and key[0] not in ('', manifest['instance_name']):
                continue
            if key[1] in displacements:
                raise ValueError('Ambiguous ML node label; manifest must identify the instance.')
            displacements[key[1]] = value
        macro = [displacements[labels[item]][0] / length for item in COMPONENTS]
        residual = 0.0
        for pair in manifest['periodic_pairs']:
            dx, dy, dz = pair['delta']
            jump = (macro[0] * dx + 0.5 * macro[3] * dy + 0.5 * macro[4] * dz,
                    0.5 * macro[3] * dx + macro[1] * dy + 0.5 * macro[5] * dz,
                    0.5 * macro[4] * dx + 0.5 * macro[5] * dy + macro[2] * dz)
            slave, master = displacements[int(pair['slave'])], displacements[int(pair['master'])]
            residual = max(residual, max(abs(slave[i] - master[i] - jump[i]) for i in range(3)))
        data = _fields(frame, mesh)
        target = _target(manifest['reference'], time)
        for actual, prescribed in zip(data['mean_stress'], target):
            error = abs(actual - prescribed)
            max_error = max(max_error, error)
            max_ratio = max(max_ratio, error / (0.01 + 1.e-6 * abs(prescribed)))
        max_periodic = max(max_periodic, residual)
        rows.append({'pseudo_time': time, 'mean_stress': data['mean_stress'],
                     'macro_strain': macro, 'mean_strain': data['mean_strain'],
                     'target_stress': target, 'periodic_residual_mm': residual})
        last_data = data
    if len(rows) < 2 or abs(rows[0]['pseudo_time']) > 1.e-12:
        raise ValueError('ML history must include the zero origin and final loading state.')
    if any(rows[i + 1]['pseudo_time'] <= rows[i]['pseudo_time'] for i in range(len(rows) - 1)):
        raise ValueError('ML frame times must be strictly increasing.')
    if max_ratio > 1.0:
        raise ValueError('ML prescribed stress check failed; maximum error {0:g} MPa.'.format(max_error))
    if max_periodic > 1.e-7:
        raise ValueError('ML periodic displacement residual exceeds 1e-7 mm.')
    checks = {'completed': True, 'max_stress_error_MPa': max_error,
              'max_stress_tolerance_ratio': max_ratio, 'max_periodic_residual_mm': max_periodic,
              'stress_absolute_tolerance_MPa': 0.01, 'stress_relative_tolerance': 1.e-6,
              'periodic_tolerance_mm': 1.e-7, 'final_time': rows[-1]['pseudo_time']}
    return rows, checks, _snapshot(odb, source, step_name, len(frames) - 1, mesh, last_data)


def _cpfem_snapshot(odb, source, manifest, config, mesh):
    reference = manifest['reference']
    target_strain, target_stress = reference['strain'][-1], reference['stress'][-1]
    strain_measure = config.get('cpfem_strain_measure', 'green')
    stress_measure = config.get('cpfem_stress_measure', 'kirchhoff')
    if strain_measure not in ('green', 'small') or stress_measure not in ('kirchhoff', 'cauchy'):
        raise ValueError('Supported reference measures: green/small strain and kirchhoff/cauchy stress.')
    strain_key = 'macro_green_strain' if strain_measure == 'green' else 'macro_strain'
    requested_step = config.get('cpfem_step')
    if requested_step is not None and sys.version_info[0] < 3:
        requested_step = requested_step.encode('utf-8')
    names = [requested_step] if requested_step else list(odb.steps.keys())
    explicit = config.get('cpfem_frame')
    if explicit is not None and requested_step is None:
        names = names[-1:]
    candidates = []
    for name in names:
        frames = odb.steps[name].frames
        indices = [int(explicit) % len(frames)] if explicit is not None else range(len(frames))
        if explicit is not None and not -len(frames) <= int(explicit) < len(frames):
            raise ValueError('Requested CPFEM frame is outside the selected step.')
        for index in indices:
            frame = frames[index]
            if 'U' not in frame.fieldOutputs or 'S' not in frame.fieldOutputs:
                continue
            kinematics = _kinematics(frame, mesh)
            errors = [abs(x - y) / (1.e-7 + 5.e-5 * abs(y))
                      for x, y in zip(kinematics[strain_key], target_strain)]
            candidates.append((max(errors), name, index, kinematics))
    if not candidates:
        raise ValueError('No CPFEM frame contains both displacement and stress.')
    candidates.sort(key=lambda item: item[0])
    error, step_name, index, kinematics = candidates[0]
    if error > 1.0:
        raise ValueError('CPFEM ODB does not match the reference endpoint strain; '
                         'closest tolerance ratio {0:g}. Check case, step, and strain measure.'.format(error))
    # Repeated strains can occur on unloading or in later steps. Check stress
    # before accepting a candidate instead of rejecting after the first match.
    best_stress_error = float('inf')
    matched = False
    for error, step_name, index, kinematics in candidates:
        if error > 1.0:
            break
        data = _fields(odb.steps[step_name].frames[index], mesh)
        factor = kinematics['macro_jacobian'] if stress_measure == 'kirchhoff' else 1.0
        reconstructed = [factor * x for x in data['mean_stress']]
        stress_errors = [abs(x - y) for x, y in zip(reconstructed, target_stress)]
        stress_ratios = [e / (0.01 + 1.e-3 * abs(y)) for e, y in zip(stress_errors, target_stress)]
        best_stress_error = min(best_stress_error, max(stress_errors))
        if max(stress_ratios) <= 1.0:
            matched = True
            break
    if not matched:
        raise ValueError('CPFEM endpoint strain matches, but no matching-strain frame has '
                         'volume-average stress reproducing the reference case '
                         '(best maximum component error {0:g} MPa).'.format(best_stress_error))
    result = _snapshot(odb, source, step_name, index, mesh, data)
    match = {'method': 'Explicit frame' if explicit is not None else 'Closest matching corner-derived endpoint strain',
             'step': step_name, 'frame_index': index, 'frame_time': result['frame_time'],
             'reference_strain_measure': strain_measure, 'reference_stress_measure': stress_measure,
             'max_strain_error': max(abs(x - y) for x, y in zip(kinematics[strain_key], target_strain)),
             'max_strain_tolerance_ratio': error, 'max_stress_error_MPa': max(stress_errors),
             'max_stress_tolerance_ratio': max(stress_ratios),
             'reference_stress': target_stress, 'reconstructed_reference_stress': reconstructed,
             'reference_strain': target_strain, 'reconstructed_reference_strain': kinematics[strain_key]}
    result['selection'] = match
    return result, match


def _odb_worker(config_path):
    with open(config_path) as handle:
        config = json.load(handle)
    with open(config['manifest_path']) as handle:
        manifest = json.load(handle)
    work_dir = os.path.abspath(config['work_dir'])
    ml = _open_odb(config['ml_odb'], work_dir, 'ml')
    try:
        history, checks, ml_snapshot = _ml_rows(ml, manifest, _mesh(ml), config['ml_odb'])
    finally:
        ml.close()
    print('ML loading, completion, and periodicity checks passed; matching CPFEM frame.')
    sys.stdout.flush()
    cpfem = _open_odb(config['cpfem_odb'], work_dir, 'cpfem')
    try:
        cpfem_snapshot, match = _cpfem_snapshot(cpfem, config['cpfem_odb'], manifest, config, _mesh(cpfem))
    finally:
        cpfem.close()
    result = {'cpfem_snapshot': cpfem_snapshot, 'ml_snapshot': ml_snapshot,
              'ml_history': history, 'cpfem_match': match, 'ml_checks': checks}
    output = os.path.join(work_dir, 'odb_results.json')
    with open(output, 'w') as handle:
        json.dump(result, handle, allow_nan=False)
    print('Matched CPFEM step {0}, frame {1}; extracted both cubes.'.format(match['step'], match['frame_index']))
    return 0


if __name__ == '__main__':
    sys.exit(_odb_worker(sys.argv[1]))
'''


def resolve_cpfem_odb(reference_path, case_key, supplied):
    """Averaged JSON histories alone cannot supply grain-resolved cube fields."""
    if supplied is not None:
        candidates = [supplied]
    else:
        candidates = [reference_path.with_suffix('.odb'),
                      reference_path.parent / (case_key + '.odb'),
                      reference_path.parent / 'inputs' / (case_key + '_Abaqus_Input_File.odb')]
        if reference_path.resolve() == DEFAULT_REFERENCE.resolve() and case_key == DEFAULT_CASE:
            candidates.append(DEFAULT_CPFEM_ODB)
    for path in candidates:
        if path.is_file():
            with path.open('rb') as stream:
                header = stream.read(128)
            if header.startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError('CPFEM ODB is a Git LFS pointer, not the downloaded data: {}. '
                                 'Run git lfs pull from the repository root.'.format(path))
            return path.resolve()
    raise FileNotFoundError('A matching CPFEM ODB is needed for the grain fields. '
                            'Looked for: {}. Download the reference ODB (git lfs pull for '
                            'an LFS checkout), or pass --cpfem-odb /path/to/reference.odb.'
                            .format(', '.join(str(path) for path in candidates)))


def run_command(command, work_dir, description):
    """Keep solver byproducts and logs in this run's isolated working directory."""
    print(description, flush=True)
    log_path = work_dir / 'run.log'
    with log_path.open('a') as stream:
        stream.write('\n' + description + '\n')
        stream.flush()
        completed = subprocess.run(command, cwd=work_dir, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        tail = '\n'.join(log_path.read_text(errors='replace').splitlines()[-18:])
        raise RuntimeError('{} failed (exit {}).\n{}'.format(description, completed.returncode, tail))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a periodic ML UMAT cube against one CPFEM case and save exactly two PNGs.',
        epilog='Default run uses the first random-texture test case and the existing trained ML material. Requires Abaqus '
               'and a Fortran compiler on PATH, plus NumPy and Matplotlib in project Python. '
               'The original CPFEM ODB is read only. This replays an existing CPFEM reference; '
               'it does not generate a new grain-resolved CPFEM simulation.')
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE,
                        help='CPFEM JSON containing Results.Sij/Eij/Epij histories')
    parser.add_argument('--cpfem-odb', type=Path,
                        help='Matching CPFEM ODB; default reference_data/cpfem_test.odb beside this script')
    parser.add_argument('--case', help='Case key; default saved test case, or first usable case for another JSON')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='Exported ML material CSV')
    parser.add_argument('--umat', type=Path, default=DEFAULT_UMAT, help='ML UMAT Fortran source')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results',
                        help='Save the two comparison PNGs here (default: %(default)s)')
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--divisions', type=int, default=4, help='ML elements per cube edge')
    parser.add_argument('--length', type=float,
                        help='Reference cube edge in mm; default from RVE_Size metadata, or .665 for the saved test case')
    parser.add_argument('--max-equivalent-strain', type=float, default=.005,
                        help='Contiguous small-deformation reference prefix (default .005)')
    parser.add_argument('--total-shear', choices=('tensor', 'engineering'), default='tensor',
                        help='Convention of E12/E13/E23 in input JSON; Ep shear must be engineering')
    parser.add_argument('--strain-measure', choices=('green', 'small'), default='green',
                        help='Recorded reference total-strain measure, used to match its ODB frame')
    parser.add_argument('--stress-measure', choices=('kirchhoff', 'cauchy'), default='kirchhoff',
                        help='Recorded reference stress measure, used to verify its ODB frame')
    parser.add_argument('--cpfem-step', help='Original CPFEM ODB step; default search all steps for the matching state')
    parser.add_argument('--cpfem-frame', type=int,
                        help='Override automatic endpoint matching; still checked against reference')
    parser.add_argument('--deformation-scale', type=float, default=1.,
                        help='Same displacement magnification for all twelve cubes')
    parser.add_argument('--color-limit', type=float,
                        help='Optional symmetric stress range +/- this MPa for the entire cube figure')
    parser.add_argument('--abaqus', default='abaqus', help='Abaqus launcher name or absolute path')
    parser.add_argument('--keep-work', action='store_true',
                        help='Retain temporary solver files for debugging; default removes them after success')
    args = parser.parse_args()
    if args.cpus < 1 or args.divisions < 1:
        parser.error('--cpus and --divisions must be positive')
    for name in ('length', 'max_equivalent_strain', 'deformation_scale', 'color_limit'):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0.):
            parser.error('--' + name.replace('_', '-') + ' must be finite and positive')
    return args


def main():
    args = parse_args()
    for path in (args.reference, args.model, args.umat):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    abaqus = shutil.which(args.abaqus)
    if abaqus is None:
        raise RuntimeError('Abaqus is not on PATH. Configure your Abaqus/Standard environment '
                           'and compatible Fortran compiler, or pass --abaqus /path/to/abaqus.')
    case_key = args.case
    if case_key is None and args.reference.resolve() == DEFAULT_REFERENCE.resolve():
        case_key = DEFAULT_CASE
    key, metadata, reference = read_reference(args.reference, case_key, 0,
                                            args.max_equivalent_strain, args.total_shear)
    cpfem_odb = resolve_cpfem_odb(args.reference, key, args.cpfem_odb)
    if args.length is None:
        size_data = metadata.get('RVE_Size')
        if (size_data is None and args.reference.resolve() == DEFAULT_REFERENCE.resolve()
                and key == DEFAULT_CASE):
            size_data = DEFAULT_REFERENCE_LENGTH
        size = np.atleast_1d(np.asarray(size_data if size_data is not None else [], dtype=float))
        if size.size not in (1, 3) or not np.isfinite(size).all() or np.any(size <= 0.):
            raise ValueError('Reference metadata needs RVE_Size or an explicit --length.')
        if not np.allclose(size, size[0], rtol=1.e-6, atol=1.e-8):
            raise ValueError('This periodic benchmark requires a cubic reference RVE.')
        length = float(size[0])
    else:
        length = args.length
    props = np.loadtxt(args.model, delimiter=',').reshape(-1)
    if len(props) < 30 or not np.isfinite(props).all():
        raise ValueError('Invalid ML material export')
    offset = float(props[7])  # Export header: nsv,nsd,C11,C12,C44,rho,gamma,epc,...
    if offset < 0.:
        raise ValueError('Exported plastic-strain offset epc must be nonnegative')
    if max(strain_norm(np.asarray(reference['plastic_strain']))) <= offset:
        raise ValueError('Reference prefix never exceeds the model plastic-strain offset; '
                         'increase --max-equivalent-strain to compare post-offset response.')
    nodes, cells, pairs, controls = make_mesh(args.divisions, length)
    manifest = dict(job='periodic_ml', source_dataset=str(args.reference.resolve()), case_key=key,
                    source_model=str(args.model.resolve()), source_umat=str(args.umat.resolve()),
                    length_mm=length, divisions=args.divisions, node_count=len(nodes),
                    element_count=len(cells), anchor_label=1, controls=controls,
                    periodic_pairs=pairs, material_mode='ml', stress_components=list(COMPONENTS),
                    reference=reference, expected_completion_time=1.)
    # All intermediate inputs, solver files, upgraded ODB copies and caches belong
    # to this unique directory. Existing runs and reference files are untouched.
    work_dir = Path(tempfile.mkdtemp(prefix='cpfem-umat-'))
    successful = False
    print('Dataset: ' + str(args.reference.resolve()), flush=True)
    print('Reference: {} ({} samples); ML mesh: {} elements; training offset: {:g}'.format(
        key, len(reference['time']), len(cells), offset), flush=True)
    print('Temporary working directory: ' + str(work_dir), flush=True)
    try:
        os.environ.setdefault('MPLCONFIGDIR', str(work_dir / 'mpl-cache'))
        os.environ.setdefault('XDG_CACHE_HOME', str(work_dir / 'cache'))
        shutil.copyfile(args.model, work_dir / 'material.inc')
        shutil.copyfile(args.umat, work_dir / 'ml_umat.f')
        manifest_path = work_dir / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
        (work_dir / 'periodic_ml.inp').write_text(input_deck(manifest, nodes, cells, props, 'material.inc'))
        run_command([abaqus, 'job=periodic_ml', 'input=periodic_ml.inp', 'user=ml_umat.f',
                     'cpus=' + str(args.cpus), 'mp_mode=threads', 'interactive'],
                    work_dir, '1/3 Running the periodic ML UMAT cube in Abaqus...')
        status = work_dir / 'periodic_ml.sta'
        if not status.is_file() or 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' not in status.read_text().upper():
            raise RuntimeError('Abaqus did not report successful completion.')
        config = dict(cpfem_odb=str(cpfem_odb), cpfem_step=args.cpfem_step,
                      cpfem_frame=args.cpfem_frame, manifest_path=str(manifest_path),
                      ml_odb=str(work_dir / 'periodic_ml.odb'), work_dir=str(work_dir),
                      cpfem_strain_measure=args.strain_measure,
                      cpfem_stress_measure=args.stress_measure)
        config_path = work_dir / 'extract_config.json'
        config_path.write_text(json.dumps(config, indent=2))
        worker_path = work_dir / 'extract_odb.py'
        worker_path.write_text(ODB_WORKER_SOURCE)
        run_command([abaqus, 'python', str(worker_path), str(config_path)], work_dir,
                    '2/3 Reading both ODBs and checking the matching CPFEM state...')
        result = json.loads((work_dir / 'odb_results.json').read_text())
        fields = [('pseudo_time', float)] + [(prefix + c, float)
                  for prefix in ('mean_S', 'macro_E') for c in COMPONENTS]
        history = np.zeros(len(result['ml_history']), dtype=fields)
        for i, row in enumerate(result['ml_history']):
            history['pseudo_time'][i] = row['pseudo_time']
            for j, component in enumerate(COMPONENTS):
                history['mean_S' + component][i] = row['mean_stress'][j]
                history['macro_E' + component][i] = row['macro_strain'][j]
        print('3/3 Saving the two comparison figures...', flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        strain_name = {'green': 'Green–Lagrange', 'small': 'small strain'}[args.strain_measure]
        stress_name = {'kirchhoff': 'Kirchhoff', 'cauchy': 'Cauchy'}[args.stress_measure]
        measures_note = ('CPFEM: {} stress / {} strain; ML UMAT: Cauchy stress / small strain'.format(
            stress_name, strain_name))
        # Render in the temporary directory first, so a plotting failure does not
        # replace one of the user's previous successful figures prematurely.
        plot_components(reference, history, offset, work_dir / 'components_shifted.png',
                        measures_note=measures_note)
        plot_cubes(result['cpfem_snapshot'], result['ml_snapshot'], work_dir / 'cubes.png',
                   deformation_scale=args.deformation_scale, color_limit=args.color_limit)
        for name in ('components_shifted.png', 'cubes.png'):
            destination = args.output_dir.resolve() / name
            shutil.copyfile(work_dir / name, destination)
            print(destination, flush=True)
        successful = True
    finally:
        if successful and not args.keep_work:
            shutil.rmtree(work_dir)
        else:
            print('Working files retained at ' + str(work_dir), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)

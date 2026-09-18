#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the CPFEM-based ML flow rule and save it for Python and Abaqus UMAT use.

This script writes:

* a pickled pyLabFEA Material for Python reuse
* Abaqus UMAT CSV/JSON SVM parameters under examples/UMAT/models

The default export is the CPFEM V2 model used by the UMAT examples:
ML-CPFEM-Random-Texture-cpfem with 15 SVM features.
"""

from __future__ import annotations

import argparse
import getpass
import json
import platform
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pylabfea as FE


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_TRAIN_DATA = SCRIPT_DIR / "Data_Random_Texture.json"
DEFAULT_TEST_DATA = SCRIPT_DIR / "Data_Random_Texture_Test.json"
DEFAULT_PYTHON_MODEL_DIR = SCRIPT_DIR / "models"
DEFAULT_UMAT_MODEL_DIR = REPO_ROOT / "examples" / "UMAT" / "models"
DEFAULT_MATERIAL_NAME = "ML-CPFEM-Random-Texture-cpfem"
DEFAULT_UMAT_FEATURES = 15


def pylabfea_version() -> str:
    try:
        return version("pylabfea")
    except PackageNotFoundError:
        return getattr(FE, "__version__", "unknown")


def check_overwrite(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing model file(s):\n"
            f"{names}\n"
            "Pass --overwrite if you want to replace them."
        )


def export_umat_parameters(
    material: FE.Material,
    output_dir: Path,
    script_path: Path,
    source_path: Path,
    feature_count: int,
    overwrite: bool,
    constant_feature_tol: float,
    training_parameters: dict[str, float | int],
) -> tuple[Path, Path]:
    """Write Abaqus-readable SVM parameters for the trained material."""

    if not material.ML_yf:
        raise AttributeError("No ML flow rule is defined on the material.")

    support_vectors = material.svm_yf.support_vectors_
    if support_vectors.shape[1] < feature_count:
        raise ValueError(
            f"Trained SVC has only {support_vectors.shape[1]} features; "
            f"cannot export {feature_count}."
        )
    if support_vectors.shape[1] > feature_count:
        extra_features = support_vectors[:, feature_count:]
        if not np.allclose(extra_features, 0.0, atol=constant_feature_tol):
            raise ValueError(
                "The trained model has non-zero features beyond the UMAT export "
                f"width ({feature_count}). Exporting would change the model."
            )
        support_vectors = support_vectors[:, :feature_count]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"abq_{material.name}-svm"
    csv_path = output_dir / f"{stem}.csv"
    meta_path = output_dir / f"{stem}_meta.json"
    check_overwrite([csv_path, meta_path], overwrite)

    dual_coef = material.svm_yf.dual_coef_[0]
    nsv = len(dual_coef)
    nlin = int((nsv * (feature_count + 1) + 30) / 8) + 1
    ndata = nlin * 8
    props = np.zeros(ndata)
    props[0] = nsv
    props[1] = feature_count
    props[2] = material.C11
    props[3] = material.C12
    props[4] = material.C44
    props[5] = material.svm_yf.intercept_[0]
    props[6] = material.gam_yf
    props[7] = material.epc
    props[8] = material.scale_seq
    props[9] = material.scale_wh

    if material.CV is None:
        props[10:16] = -1.0
    else:
        props[10] = material.CV[1, 1]
        props[11] = material.CV[2, 2]
        props[12] = material.CV[0, 2]
        props[13] = material.CV[1, 2]
        props[14] = material.CV[4, 4]
        props[15] = material.CV[5, 5]

    nset = 1 if material.Nset is None else material.Nset
    scale_text = material.scale_text
    if scale_text is None:
        scale_text = np.ones(nset)

    props[16] = -1.0 if material.dev_only else 0.0
    props[17] = nset
    props[18:18 + nset] = scale_text
    props[29:29 + nsv] = dual_coef
    last = 29 + nsv + feature_count * nsv
    props[29 + nsv:last] = support_vectors.flatten()

    np.savetxt(csv_path, props.reshape((nlin, 8)), delimiter=", ", newline="\n")

    descr = list(training_parameters.keys())
    param = list(training_parameters.values())
    descr.extend(["Ndata", "gamma", "C", "UMAT_features"])
    param.extend([ndata, material.gam_yf, material.C_yf, feature_count])

    sys_info = platform.uname()
    meta = {
        "Info": {
            "Owner": getpass.getuser(),
            "Institution": "ICAMS, Ruhr University Bochum, Germany",
            "Date": str(date.today()),
            "Description": "CPFEM-trained SVC parameters for ML plasticity UMAT",
            "Method": "Support Vector Classification",
            "System": {
                "sysname": sys_info.system,
                "nodename": sys_info.node,
                "release": sys_info.release,
                "version": sys_info.version,
                "machine": sys_info.machine,
            },
        },
        "Model": {
            "Creator": "pylabfea",
            "Version": pylabfea_version(),
            "Repository": "https://github.com/AHartmaier/pyLabFEA.git",
            "Input": str(source_path),
            "Script": str(script_path),
            "Names": descr,
            "Parameters": param,
        },
        "Data": {
            "Class": "SVC_parameters",
            "Type": "CSV",
            "File": str(csv_path),
            "Separator": ",",
            "Header": None,
            "Format": [nlin, 8],
            "Names": [
                "nsv",
                "nsd",
                "C11",
                "C12",
                "C44",
                "rho",
                "gamma",
                "epc",
                "scale_seq",
                "scale_wh",
                "C22",
                "C33",
                "C13",
                "C23",
                "C55",
                "C66",
                "Nset",
                "scale_text[0:Nset]",
                "dual_coef[0:nsv]",
                "sup_vec[0:nsv,0:nsd]",
            ],
            "Units": {
                "Stress": "MPa",
                "Strain": "None",
                "Disp": "mm",
                "Force": "N",
            },
        },
    }
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    return csv_path, meta_path


def train_material(args: argparse.Namespace) -> FE.Material:
    db = FE.Data(
        str(args.train_data),
        mat_name=args.material_name,
        epl_crit=args.epl_crit,
        epl_start=args.epl_start,
        epl_max=args.epl_max,
        depl=args.depl,
        wh_data=True,
    )
    print(f"Imported {db.mat_data['Nlc']} CPFEM load cases.")

    material = FE.Material(db.mat_data["Name"], num=1)
    material.from_data(db.mat_data)
    material.train_SVC(
        C=args.C,
        gamma=args.gamma,
        Fe=args.Fe,
        Ce=args.Ce,
        Nseq=args.Nseq,
        gridsearch=False,
        plot=False,
    )
    print(f"Training complete. Support vectors: {len(material.svm_yf.support_vectors_)}")
    return material


def save_python_material(material: FE.Material, output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = output_dir / f"mat_{material.name}.pkl"
    check_overwrite([pickle_path], overwrite)
    material.pckl(path=str(output_dir))
    return pickle_path


def validate_model(material: FE.Material, test_data: Path) -> None:
    sig_tot, epl_tot, yf_ref = FE.create_test_sig(file=str(test_data))
    yf_ml = material.calc_yf(sig_tot, epl_tot, pred=False)
    FE.training_score(yf_ref, yf_ml)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CPFEM ML plasticity and save Python plus UMAT model files."
    )
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--material-name", default=DEFAULT_MATERIAL_NAME)
    parser.add_argument("--python-model-dir", type=Path, default=DEFAULT_PYTHON_MODEL_DIR)
    parser.add_argument("--umat-model-dir", type=Path, default=DEFAULT_UMAT_MODEL_DIR)
    parser.add_argument("--C", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--Fe", type=float, default=0.7)
    parser.add_argument("--Ce", type=float, default=0.9)
    parser.add_argument("--Nseq", type=int, default=2)
    parser.add_argument("--epl-crit", type=float, default=2.0e-3)
    parser.add_argument("--epl-start", type=float, default=1.0e-3)
    parser.add_argument("--epl-max", type=float, default=0.03)
    parser.add_argument("--depl", type=float, default=1.0e-3)
    parser.add_argument("--umat-features", type=int, default=DEFAULT_UMAT_FEATURES)
    parser.add_argument("--constant-feature-tol", type=float, default=1.0e-12)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    material = train_material(args)

    pickle_path = save_python_material(material, args.python_model_dir, args.overwrite)
    training_parameters = {
        "Fe": args.Fe,
        "Ce": args.Ce,
        "Nseq": args.Nseq,
        "epl_crit": args.epl_crit,
        "epl_start": args.epl_start,
        "epl_max": args.epl_max,
        "depl": args.depl,
    }
    csv_path, meta_path = export_umat_parameters(
        material=material,
        output_dir=args.umat_model_dir,
        script_path=Path(__file__).resolve(),
        source_path=args.train_data.resolve(),
        feature_count=args.umat_features,
        overwrite=args.overwrite,
        constant_feature_tol=args.constant_feature_tol,
        training_parameters=training_parameters,
    )

    if not args.skip_validation:
        validate_model(material, args.test_data)

    print("Saved Python material:", pickle_path)
    print("Saved UMAT CSV:", csv_path)
    print("Saved UMAT metadata:", meta_path)
    print("Abaqus material name:", material.name)


if __name__ == "__main__":
    main()

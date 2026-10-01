"""Generate the fixed kt-Gaussian undersampling masks of the validation and
test cases of a split manifest, for R = 10, 20, 30, 40, 50.

Masks are written as usmask_ktGaussian{R}.{zarr,h5,mat} with the challenge's
(1, Nt, 1, SPE, PE, 1) layout into the case directories below --output_root.

Run with:  uv run python -m data.create_test_val_umasks --data_root <.../TaskR1R2/TrainSet>
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec

from data.data_utils import derive_compressed_root, h5_first_key, load_csv_params
from data.ktgaussian import generate_ktgaussian_mask

ACCELERATIONS = (10, 20, 30, 40, 50)


def process_case(case_dict, data_root, output_root, acceleration_factors, formats):
    case_path = data_root / case_dict["path"]
    out_case_path = output_root / case_dict["path"]
    kdata_file = case_path / "kdata_full.mat"
    params_file = case_path / "params.csv"

    if not case_path.exists():
        print(f"Skipping {case_path}: Directory not found.")
        return

    out_case_path.mkdir(parents=True, exist_ok=True)

    # Dimensions from params.csv, where matrix_size holds the kdata shape
    # (Nv, Nt, Nc, SPE, PE, FE), e.g. [4, 23, 10, 33, 107, 144].
    pe_dim, spe_dim, nt = 0, 0, 0
    if params_file.exists():
        params = load_csv_params(params_file)
        matrix_size = params.get("matrix_size", None)
        if matrix_size and isinstance(matrix_size, list) and len(matrix_size) >= 5:
            pe_dim = int(matrix_size[-2])
            spe_dim = int(matrix_size[-3])
            nt = int(matrix_size[1])

    # Otherwise read the shape from the kdata_full.mat header.
    if pe_dim == 0 or spe_dim == 0 or nt == 0:
        if not kdata_file.exists():
            print(
                f"Skipping {case_path}: kdata_full.mat not found for fallback dimension check."
            )
            return

        with h5py.File(kdata_file, "r") as f:
            _, nt, _, spe_dim, pe_dim, _ = f[h5_first_key(f)].shape

    for R in acceleration_factors:
        mask_np = generate_ktgaussian_mask(spe_dim, pe_dim, nt, R)
        # (Nt, SPE, PE) -> (1, Nt, 1, SPE, PE, 1)
        mask_np = mask_np.reshape(1, nt, 1, spe_dim, pe_dim, 1).astype(np.uint8)

        base_filename = f"usmask_ktGaussian{R}"

        if "mat" in formats:
            mat_path = out_case_path / f"{base_filename}.mat"
            with h5py.File(mat_path, "w") as f:
                f.create_dataset(base_filename, data=mask_np, compression="gzip")

        if "h5" in formats:
            h5_path = out_case_path / f"{base_filename}.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset(base_filename, data=mask_np, compression="lzf")

        if "zarr" in formats:
            zarr_path = out_case_path / f"{base_filename}.zarr"
            compressor = BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")
            root = zarr.open(str(zarr_path), mode="w")

            arr = root.create_array(
                base_filename,
                shape=mask_np.shape,
                dtype=mask_np.dtype,
                compressors=[compressor],
            )
            arr[:] = mask_np

        print(
            f"Generated masks for {out_case_path} at R={R} with shapes {mask_np.shape} in formats: {formats}"
        )


def main(args):
    data_root = Path(args.data_root)
    output_root = Path(
        args.output_root
        if args.output_root is not None
        else derive_compressed_root(str(data_root))
    )
    yaml_path = Path(args.yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Split configuration not found at {yaml_path}")

    with open(yaml_path, "r") as f:
        splits = yaml.safe_load(f)

    target_cases = []

    if "test_set" in splits:
        target_cases.extend(splits["test_set"])

    if "folds" in splits:
        for fold_key, fold_data in splits["folds"].items():
            if "val" in fold_data:
                target_cases.extend(fold_data["val"])

    unique_targets = {c["path"]: c for c in target_cases}.values()
    print(f"Found {len(unique_targets)} artificial val/test cases to process.")
    print(f"Writing masks to: {output_root}")

    for case_dict in tqdm(unique_targets, desc="Generating Masks"):
        process_case(
            case_dict=case_dict,
            data_root=data_root,
            output_root=output_root,
            acceleration_factors=ACCELERATIONS,
            formats=args.formats,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic undersampling masks for split data."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Raw training data, e.g. .../ChallengeData/TaskR1R2/TrainSet.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output root, mirroring the case directories of --data_root. "
        "Defaults to --data_root with its 'TaskR1R2' path segment replaced by "
        "'TaskR1R2_compressed'.",
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="data/dataset_splits.yaml",
        help="Split manifest whose val/test cases get masks.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["mat", "h5", "zarr"],
        default=["zarr"],
        help="Output format(s); must include the data_format used for training "
        "(zarr by default).",
    )
    args = parser.parse_args()

    main(args)

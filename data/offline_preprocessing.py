"""Offline preprocessing of the raw CMRx4DFlow training data.

For every case directory below --data_root (identified by its params.csv) the
fully sampled k-space, coil maps and segmentation are converted into a single
chunked, compressed Zarr (default) or HDF5 store, mirrored below --output_root:

    <output_root>/<case path>/data_compressed[_nc<N>].zarr

Run with:  uv run python -m data.offline_preprocessing --data_root <.../TaskR1R2/TrainSet>
"""

import argparse
import os
import shutil
from pathlib import Path

import h5py
import torch
import zarr
from tqdm import tqdm
from zarr.codecs import BloscCodec

from data.data_utils import derive_compressed_root, load_csv_params, load_mat
from utils.sense import centered_ifftn

# ==============================================================================
# OUTPUT SPECIFICATION
# ==============================================================================
# Note that the coil (Nc) and cardiac phase (Nt) axes are swapped compared to the
# raw MATLAB .mat inputs.
#
# 1. "data_x"
#    - Shape: (Nv, Nc_out, Nt, SPE, PE, FE)
#      where Nc_out = target_nc (if compress_coils=True) else original Nc.
#    - Space: Hybrid Space (kz-ky-x domain)
#      * A centred 1D inverse FFT is applied along the readout axis (FE).
#      * SPE (Z-axis) and PE (Y-axis) remain in k-space.
#      * FE (X-axis) is transformed into Image Space.
#    - Data Type: Complex64
#
# 2. "smaps"
#    - Shape: (Nc_out, SPE, PE, FE)
#      where Nc_out = target_nc (if compress_coils=True) else original Nc.
#    - Space: Image Space
#      * Pre-computed complex coil sensitivity maps normalized using the
#        Sum-of-Squares (SoS) L2 norm across the coil dimension.
#    - Data Type: Complex64
#
# 3. "segmask"
#    - Shape: (SPE, PE, FE)
#    - Space: Image Space
#      * Binary Region of Interest (ROI) mask mapping the target vasculature.
#    - Data Type: Boolean
#
# 4. Attributes (f.attrs)
#    - Metadata parameters parsed from params.csv stored as key-value pairs.
# ==============================================================================


def _compress_coils(kdata, coilmap, target_nc):
    """SVD coil compression of kdata (Nv, Nt, Nc, SPE, PE, FE) and the coil maps
    to the ``target_nc`` dominant virtual coils. Returns kdata as
    (Nv, target_nc, Nt, SPE, PE, FE)."""
    Nv, Nt, Nc, SPE, PE, FE = kdata.shape
    if Nc <= target_nc:
        return kdata.permute(0, 2, 1, 3, 4, 5), coilmap

    kdata_flat = kdata.permute(2, 0, 1, 3, 4, 5).reshape(Nc, -1)
    cov = kdata_flat @ kdata_flat.conj().T
    _, V = torch.linalg.eigh(cov)
    proj = V[:, -target_nc:]

    kdata_res = (kdata_flat.T @ proj).T
    kdata_compressed = kdata_res.view(target_nc, Nv, Nt, SPE, PE, FE).permute(
        1, 0, 2, 3, 4, 5
    )

    cmap_flat = coilmap.reshape(Nc, -1)
    cmap_res = (cmap_flat.T @ proj).T
    coilmap_compressed = cmap_res.view(target_nc, SPE, PE, FE)

    return kdata_compressed, coilmap_compressed


def run_preprocessing(
    data_root,
    output_root=None,
    target_nc=5,
    overwrite=False,
    compress_coils=False,
    output_format="zarr",
):
    data_root_abs = os.path.abspath(data_root)
    print(f"Scanning directory tree: {data_root_abs}")

    # By default the output mirrors data_root below a sibling
    # "TaskR1R2_compressed" tree (see derive_compressed_root).
    output_root_abs = os.path.abspath(
        output_root
        if output_root is not None
        else derive_compressed_root(data_root_abs)
    )
    print(f"Writing preprocessed output tree to: {output_root_abs}")

    case_dirs = []
    for root, dirs, files in os.walk(data_root_abs):
        if "params.csv" in files:
            case_dirs.append(root)

    print(
        f"Found {len(case_dirs)} cases. Starting preprocessing "
        f"(format={output_format}, target_nc={target_nc}, overwrite={overwrite}, compress_coils={compress_coils})..."
    )

    if len(case_dirs) == 0:
        print(f"ERROR: Could not find any 'params.csv' files inside {data_root_abs}")
        return

    for case_dir in tqdm(case_dirs):
        case_path = Path(case_dir)
        rel_case_dir = os.path.relpath(case_dir, data_root_abs)
        out_case_dir = Path(output_root_abs) / rel_case_dir
        out_case_dir.mkdir(parents=True, exist_ok=True)

        ext = "zarr" if output_format == "zarr" else "h5"
        suffix = f"_nc{target_nc}" if compress_coils else ""
        out_path = out_case_dir / f"data_compressed{suffix}.{ext}"

        if out_path.exists():
            if overwrite:
                if output_format == "zarr":
                    shutil.rmtree(out_path)
                else:
                    out_path.unlink()
            else:
                continue

        kdata_full = load_mat(case_path / "kdata_full.mat", as_complex=True)
        smaps = load_mat(case_path / "coilmap.mat", as_complex=True)
        segmask = load_mat(case_path / "segmask.mat")
        params = load_csv_params(case_path / "params.csv")

        if compress_coils:
            kdata_comp, smaps_comp = _compress_coils(kdata_full, smaps, target_nc)
        else:
            kdata_comp = kdata_full.permute(0, 2, 1, 3, 4, 5)
            smaps_comp = smaps

        sos_norm = torch.sqrt(
            torch.sum(torch.abs(smaps_comp) ** 2, dim=0, keepdim=True) + 1e-8
        )
        smaps_comp = smaps_comp / sos_norm
        segmask_comp = segmask.to(torch.bool)

        # Readout axis to image space (hybrid kz-ky-x layout).
        data_x_comp = centered_ifftn(kdata_comp, dim=(-1,))

        data_shape = data_x_comp.shape
        optimal_chunk_shape = (
            1,
            data_shape[1],
            min(8, data_shape[2]),
            data_shape[3],
            data_shape[4],
            min(16, data_shape[5]),
        )

        if output_format == "zarr":
            # Zarr v3 Native LZ4 compression
            compressor = BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")
            root = zarr.open(out_path, mode="w")

            data_x_np = data_x_comp.numpy()
            arr_x = root.create_array(
                "data_x",
                shape=data_x_np.shape,
                chunks=optimal_chunk_shape,
                dtype=data_x_np.dtype,
                compressors=[compressor],
            )
            arr_x[:] = data_x_np

            smaps_np = smaps_comp.numpy()
            arr_smaps = root.create_array(
                "smaps",
                shape=smaps_np.shape,
                dtype=smaps_np.dtype,
                compressors=[compressor],
            )
            arr_smaps[:] = smaps_np

            segmask_np = segmask_comp.numpy()
            arr_segmask = root.create_array(
                "segmask",
                shape=segmask_np.shape,
                dtype=segmask_np.dtype,
                compressors=[compressor],
            )
            arr_segmask[:] = segmask_np

            # Write attributes
            root.attrs.update(params)

        elif output_format == "h5":
            # Standard LZF compressed HDF5
            with h5py.File(out_path, "w") as f:
                f.create_dataset(
                    "data_x",
                    data=data_x_comp.numpy(),
                    chunks=optimal_chunk_shape,
                    compression="lzf",
                )
                f.create_dataset(
                    "smaps", data=smaps_comp.numpy(), chunks=True, compression="lzf"
                )
                f.create_dataset(
                    "segmask", data=segmask_comp.numpy(), chunks=True, compression="lzf"
                )

                for k, v in params.items():
                    f.attrs[k] = v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline preprocessing for 4D Flow MRI."
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
        "--format",
        type=str,
        choices=["h5", "zarr"],
        default="zarr",
        help="Output storage format (h5 or zarr).",
    )
    parser.add_argument(
        "--target_nc",
        type=int,
        default=5,
        help="Number of virtual coils kept by --compress_coils.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--compress_coils",
        action="store_true",
        help="SVD-compress the coils to --target_nc virtual coils.",
    )

    args = parser.parse_args()

    run_preprocessing(
        args.data_root,
        output_root=args.output_root,
        target_nc=args.target_nc,
        overwrite=args.overwrite,
        compress_coils=args.compress_coils,
        output_format=args.format,
    )

"""
Inference on data without ground truth (e.g. the official CMRx4DFlow
ValidationSet / TestSet).

Runs a trained checkpoint over every case found below --input_dir (see
data/realdataset.py) and writes one sparse-COO .npz per case and acceleration
factor, mirroring the input directory layout:

    {output_dir}/<case path relative to input_dir>/img_ktGaussian{R}.npz

The reconstruction is rescaled to the physical intensity scale and set to zero
outside the segmentation mask before saving.

Example:
    uv run predict.py         --ckpt_path checkpoints/flowmodl_seed42_best.pth         --model_config configs/model/flowmodl.yaml         --input_dir /data/ChallengeData/TaskR1R2/ValidationSet         --output_dir predictions
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.realdataset import RealChallengeDataset
from utils.inference import load_model, reconstruct
from utils.npz_io import save_coo_npz


@torch.inference_mode()
def timed_reconstruct(model, kspace_us, mask, smaps, usrate, device, **flags):
    """Final reconstruction (B, Nv, Nt, SPE, PE, FE) and its wall-clock time."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred = reconstruct(model, kspace_us, mask, smaps, usrate=usrate, **flags)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return pred, time.perf_counter() - t0


def run(args):
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("WARNING: running on CPU, reconstruction will be slow.")

    # Case-dependent volume shapes make cuDNN autotuning counterproductive.
    torch.backends.cudnn.benchmark = False

    model, per_encoding, decouple_readout = load_model(
        args.ckpt_path, args.model_config, device
    )
    print(f"Loaded {args.model_config} from {args.ckpt_path}")

    dataset = RealChallengeDataset(
        root_dirs=args.input_dir,
        accelerations=args.accelerations,
        in_base_dir=args.in_base_dir,
    )
    print(f"Found {len(dataset)} (case, R) items under {args.input_dir}")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "predict_log.csv"
    log_rows = []
    total_elapsed = 0.0
    n_done, n_skipped, n_failed = 0, 0, 0
    run_t0 = time.perf_counter()

    for i, batch in enumerate(loader):
        case_dir = batch["case_dir"][0]
        rel_path = batch["out_rel_path"][0]
        R = int(batch["R"][0])

        case_out_dir = output_dir / rel_path
        out_file = case_out_dir / f"img_ktGaussian{R}.npz"

        if out_file.exists() and not args.overwrite:
            print(f"[{i+1}/{len(dataset)}] SKIP (exists) {out_file}")
            n_skipped += 1
            continue

        try:
            kspace_us = batch["kspace_us"].to(device, non_blocking=True)
            mask = batch["undersampling_mask"].to(device, non_blocking=True)
            smaps = batch["sensitivity_maps"].to(device, non_blocking=True)
            segmask = batch["segmask"].to(device, non_blocking=True)
            norm = batch["norm"].to(device, non_blocking=True)
            usrate = batch["usrate_true"].to(device, non_blocking=True)

            final_pred, elapsed = timed_reconstruct(
                model,
                kspace_us,
                mask,
                smaps,
                usrate,
                device,
                per_encoding=per_encoding,
                decouple_readout=decouple_readout,
            )

            # Undo the k-space normalisation and zero everything outside the ROI.
            final_pred = final_pred * norm.view(-1, 1, 1, 1, 1, 1)
            seg = segmask.to(final_pred.dtype)[:, None, None, :, :, :]
            final_pred = final_pred * seg

            img = final_pred[0].cpu().numpy().astype(np.complex64)  # (Nv, Nt, SPE, PE, FE)

            case_out_dir.mkdir(parents=True, exist_ok=True)
            save_coo_npz(str(out_file), img)

            total_elapsed += elapsed
            n_done += 1
            print(
                f"[{i+1}/{len(dataset)}] {rel_path} R={R} "
                f"shape={img.shape} recon={elapsed:.2f}s -> {out_file}"
            )
            log_rows.append([case_dir, R, elapsed, "ok"])

        except Exception as e:
            # Log the failure and continue with the remaining cases.
            n_failed += 1
            print(f"[{i+1}/{len(dataset)}] FAILED {case_dir} R={R}: {e}")
            log_rows.append([case_dir, R, None, f"error: {e}"])

    wall_elapsed = time.perf_counter() - run_t0

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_dir", "R", "recon_seconds", "status"])
        w.writerows(log_rows)

    print(
        f"\nDone. {n_done} reconstructed, {n_skipped} skipped (already existed), "
        f"{n_failed} failed, out of {len(dataset)} total."
    )
    print(
        f"Summed reconstruction time: {total_elapsed:.1f}s; "
        f"wall-clock: {wall_elapsed:.1f}s."
    )
    print(f"Per-case log written to {log_path}")

    if n_failed > 0:
        raise SystemExit(
            f"{n_failed} case(s) failed to reconstruct -- see {log_path}. "
            "Rerun without --overwrite to retry only the missing cases."
        )


def parse_args():
    p = argparse.ArgumentParser(description="Inference on data without ground truth.")
    p.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Trained checkpoint (state_dict .pth).",
    )
    p.add_argument(
        "--model_config",
        type=str,
        default="configs/model/flowmodl.yaml",
        help="Model config matching the checkpoint.",
    )
    p.add_argument(
        "--input_dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories to scan recursively for cases.",
    )
    p.add_argument(
        "--in_base_dir",
        type=str,
        default=None,
        help="Base dir used to compute each case's relative output path. Defaults to input_dir "
        "(or their common parent if multiple).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the reconstructions to.",
    )
    p.add_argument(
        "--accelerations",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to these acceleration factors (default: all present per case).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Defaults to cuda if available, else cpu.",
    )
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Recompute cases whose output .npz already exists (default: skip them, for resumability).",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

"""
Evaluate the classical baselines and trained checkpoints on the validation or
test cases of the split, for every acceleration factor in --accelerations.

For every model and R the script reports nRMSE, SSIM, RelErr, AngErr, the
complex-difference error and the mean reconstruction time per case, followed by
the average over all acceleration factors. As in the official CMRx4DFlow
evaluation, a smooth background phase-offset model (MSAC, utils/bgc.py) is fitted
to the ground truth and removed from both the ground truth and the reconstruction
before RelErr and AngErr are computed.

Multiple seeds: train.py names checkpoints "<run>_seed<SEED>_best.pth". Passing
any one of them also evaluates every sibling checkpoint that differs only in the
seed (as "<Model>_<SEED>"); the mean and sample standard deviation over seeds
are written to a companion summary CSV. Use --seeds to restrict the seeds or
--no_seed_expansion to evaluate only the named file.

Example:
    uv run evaluate.py \
        --root_dir /data/ChallengeData/TaskR1R2_compressed/TrainSet \
        --flowmodl_ckpt checkpoints/flowmodl_seed42_best.pth \
        --modl_ckpt checkpoints/modl_seed42_best.pth \
        --flowvn_ckpt checkpoints/flowvn_seed42_best.pth \
        --output_csv results.csv
"""

import argparse
import csv
import glob as globlib
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import CMRx4DFlowDataset
from models.baselines import CGSENSE, ZeroFilled
from utils.bgc import execute_MSAC
from utils.inference import load_model, reconstruct
from utils.metrics import (
    compute_complex_diff_err,
    compute_nrmse,
    compute_ssim,
    compute_velocity_metrics,
    extract_velocity,
)

METRIC_NAMES = ["nRMSE", "SSIM", "RelErr", "AngErr", "ComplexErr"]
AVERAGE_LABEL = "Average"

# Trained models that can be evaluated: name -> default model config.
TRAINED_MODELS = {
    "FlowMoDL": "configs/model/flowmodl.yaml",
    "MoDL": "configs/model/modl.yaml",
    "FlowVN": "configs/model/flowvn.yaml",
}

# How a seed is encoded in a checkpoint filename: "<run>_seed<SEED>_..." (as
# written by train.py) or a trailing "_<SEED>".
_SEED_TOKEN_PATTERNS = (
    re.compile(r"^(?P<pre>.*_seed)(?P<seed>\d+)(?P<post>.*)$"),
    re.compile(r"^(?P<pre>.*_)(?P<seed>\d+)(?P<post>)$"),
)


def _split_seed(stem):
    """(prefix, seed, suffix) of a checkpoint stem, or None without a seed token."""
    for pattern in _SEED_TOKEN_PATTERNS:
        match = pattern.match(stem)
        if match is not None:
            return match.group("pre"), int(match.group("seed")), match.group("post")
    return None


def _expand_seed_checkpoints(ckpt, expand=True, seeds=None):
    """[(seed, path)] of every checkpoint that differs from ``ckpt`` only in
    its seed token, sorted by seed."""
    path = Path(ckpt)
    split = _split_seed(path.stem)
    if split is None:
        if seeds is not None:
            raise ValueError(f"--seeds was given but '{path.name}' has no seed token.")
        return [(None, str(path))]

    prefix, named_seed, suffix = split
    if not expand and seeds is None:
        return [(named_seed, str(path))]

    pattern = f"{globlib.escape(prefix)}*{globlib.escape(suffix + path.suffix)}"
    found = {}
    for candidate in sorted(path.parent.glob(pattern)):
        candidate_split = _split_seed(candidate.stem)
        if candidate_split and candidate_split[0] == prefix and candidate_split[2] == suffix:
            found[candidate_split[1]] = str(candidate)
    if not found:
        raise FileNotFoundError(f"No checkpoints matching '{pattern}' in {path.parent}.")

    if seeds is not None:
        missing = sorted(set(seeds) - set(found))
        if missing:
            raise FileNotFoundError(
                f"Seed(s) {missing} not found in {path.parent}; available: {sorted(found)}."
            )
        found = {seed: found[seed] for seed in seeds}
    return sorted(found.items())


def build_models(args, device):
    """Returns {name: (model, per_encoding, decouple_readout)} and
    {name: base model name}."""
    models = {}
    base_names = {}

    if not args.no_baselines:
        for name, model in (
            ("ZeroFilled", ZeroFilled()),
            ("CG-SENSE", CGSENSE(num_iter=args.cg_iters)),
        ):
            models[name] = (model.to(device).eval(), False, False)
            base_names[name] = name

    for base_name in TRAINED_MODELS:
        key = base_name.lower()
        ckpt = getattr(args, f"{key}_ckpt")
        if ckpt is None:
            continue
        checkpoints = _expand_seed_checkpoints(
            ckpt, expand=not args.no_seed_expansion, seeds=args.seeds
        )
        multiseed = len(checkpoints) > 1
        if multiseed:
            print(f"{base_name}: seeds {', '.join(str(s) for s, _ in checkpoints)}")
        for seed, path in checkpoints:
            name = f"{base_name}_{seed}" if multiseed else base_name
            models[name] = load_model(path, getattr(args, f"{key}_config"), device)
            base_names[name] = base_name

    return models, base_names


def _timed_reconstruct(model_entry, data, device):
    model, per_encoding, decouple_readout = model_entry
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        pred = reconstruct(
            model,
            data["kspace_us"],
            data["mask"],
            data["smaps"],
            usrate=data["usrate"],
            per_encoding=per_encoding,
            decouple_readout=decouple_readout,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return pred, time.perf_counter() - t0


def _remove_background_phase(image, corr_maps):
    """Subtract the fitted background phase from the flow-encoded channels of a
    (1, Nv, Nt, Z, Y, X) image."""
    img_np = image[0].detach().cpu().numpy().copy()
    img_np[1:] *= np.exp(-1j * corr_maps)
    return torch.from_numpy(img_np).to(image.device).unsqueeze(0)


def evaluate(args):
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    models, base_names = build_models(args, device)
    print(f"Evaluating {len(models)} models: {', '.join(models)}")

    # results[model][R][metric], times[model][R], counts[model][R]
    results = {name: {} for name in models}
    times = {name: {} for name in models}
    counts = {name: {} for name in models}

    for R in args.accelerations:
        dataset = CMRx4DFlowDataset(
            root_dir=args.root_dir,
            split_yaml_path=args.split_yaml,
            mode=args.mode,
            fold=args.fold,
            data_format=args.data_format,
            val_acceleration=R,
        )
        if len(dataset) == 0:
            print(f"[R={R}] No cases found -- skipping.")
            continue
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        for name in models:
            results[name][R] = {m: 0.0 for m in METRIC_NAMES}
            times[name][R] = 0.0
            counts[name][R] = 0

        for batch in loader:
            data = {
                "kspace_us": batch["kspace_us"],
                "mask": batch["undersampling_mask"],
                "smaps": batch["sensitivity_maps"],
                "target_image": batch["target_image"],
                "segmask": batch["segmask"],
                "v_enc": batch["v_enc"],
                "usrate": batch["usrate_true"],
            }
            data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
            target_mag = data["target_image"].abs()
            segmask = data["segmask"]

            # Background phase correction, fitted once per case on the ground truth.
            corr_maps = execute_MSAC(
                data["target_image"][0].cpu().numpy(), corr_fit_order=3, th=0.1
            )
            target_velocity = extract_velocity(
                _remove_background_phase(data["target_image"], corr_maps), data["v_enc"]
            )

            for name, entry in models.items():
                pred, elapsed = _timed_reconstruct(entry, data, device)
                pred_velocity = extract_velocity(
                    _remove_background_phase(pred, corr_maps), data["v_enc"]
                )
                rel_err, ang_err = compute_velocity_metrics(
                    pred_velocity, target_velocity, segmask
                )
                acc = results[name][R]
                acc["nRMSE"] += compute_nrmse(pred.abs(), target_mag, segmask).item()
                acc["SSIM"] += compute_ssim(pred.abs(), target_mag, segmask).item()
                acc["RelErr"] += rel_err.item()
                acc["AngErr"] += ang_err.item()
                acc["ComplexErr"] += compute_complex_diff_err(
                    pred, data["target_image"], segmask
                ).item()
                times[name][R] += elapsed
                counts[name][R] += 1
                del pred, pred_velocity

        for name in models:
            n = max(counts[name][R], 1)
            for m in METRIC_NAMES:
                results[name][R][m] /= n
            times[name][R] /= n
        _print_block(R, models, results, times)

    avg_results, avg_times = _compute_averages(models, results, times, counts)
    if avg_results:
        _print_block(AVERAGE_LABEL, models, avg_results, avg_times)
        for name in avg_results:
            results[name][AVERAGE_LABEL] = avg_results[name]
            times[name][AVERAGE_LABEL] = avg_times[name]

    summary = _aggregate_seeds(models, base_names, results, times)
    _print_seed_summary(summary)

    _write_csv(args.output_csv, models, results, times)
    print(f"\nSaved per-model results to {args.output_csv}")
    summary_csv = args.summary_csv or _default_summary_csv(args.output_csv)
    _write_summary_csv(summary_csv, summary)
    print(f"Saved seed mean/std summary to {summary_csv}")


def _r_sort_key(r):
    return (r == AVERAGE_LABEL, r if r != AVERAGE_LABEL else 0)


def _compute_averages(models, results, times, counts):
    """Case-count-weighted average over all evaluated acceleration factors."""
    avg_results, avg_times = {}, {}
    for name in models:
        total_n = sum(counts[name].values())
        if total_n == 0:
            continue
        avg_results[name] = {
            m: sum(results[name][R][m] * counts[name][R] for R in counts[name]) / total_n
            for m in METRIC_NAMES
        }
        avg_times[name] = (
            sum(times[name][R] * counts[name][R] for R in counts[name]) / total_n
        )
    return avg_results, avg_times


def _aggregate_seeds(models, base_names, results, times):
    """summary[base][R] = {"n", "mean", "std", "time_mean", "time_std"} over the
    seeds of every base model (each seed counts once; sample std, 0 for n = 1)."""
    grouped = {}
    for name in models:
        grouped.setdefault(base_names[name], []).append(name)

    summary = {}
    for base, names in grouped.items():
        per_r = {}
        for R in sorted({R for n in names for R in results[n]}, key=_r_sort_key):
            members = [n for n in names if R in results[n]]
            n = len(members)

            def stats(values):
                return float(np.mean(values)), (
                    float(np.std(values, ddof=1)) if n > 1 else 0.0
                )

            metric_stats = {
                m: stats([results[x][R][m] for x in members]) for m in METRIC_NAMES
            }
            time_mean, time_std = stats([times[x][R] for x in members])
            per_r[R] = {
                "n": n,
                "mean": {m: s[0] for m, s in metric_stats.items()},
                "std": {m: s[1] for m, s in metric_stats.items()},
                "time_mean": time_mean,
                "time_std": time_std,
            }
        if per_r:
            summary[base] = per_r
    return summary


def _block_label(R):
    return "Average (all accelerations)" if R == AVERAGE_LABEL else f"R={R}"


def _print_block(R, models, results, times):
    print(f"\n=== {_block_label(R)} ===")
    header = f"{'Model':<14}" + "".join(f"{m:>12}" for m in METRIC_NAMES) + f"{'Time[s]':>12}"
    print(header)
    print("-" * len(header))
    for name in models:
        if R in results[name]:
            row = f"{name:<14}" + "".join(
                f"{results[name][R][m]:>12.4f}" for m in METRIC_NAMES
            )
            print(row + f"{times[name][R]:>12.3f}")


def _print_seed_summary(summary):
    """Mean +/- std over seeds; skipped when no model has more than one seed."""
    if not any(c["n"] > 1 for per_r in summary.values() for c in per_r.values()):
        return
    r_keys = sorted({R for per_r in summary.values() for R in per_r}, key=_r_sort_key)
    for R in r_keys:
        print(f"\n=== {_block_label(R)} -- mean +/- std over seeds ===")
        header = (
            f"{'Model':<14}{'seeds':>6}"
            + "".join(f"{m:>20}" for m in METRIC_NAMES)
            + f"{'Time[s]':>20}"
        )
        print(header)
        print("-" * len(header))
        for base, per_r in summary.items():
            if R not in per_r:
                continue
            cell = per_r[R]
            row = f"{base:<14}{cell['n']:>6}"
            for m in METRIC_NAMES:
                row += f"{cell['mean'][m]:>11.4f} +/-{cell['std'][m]:>8.4f}"
            row += f"{cell['time_mean']:>11.3f} +/-{cell['time_std']:>8.3f}"
            print(row)


def _write_csv(path, models, results, times):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "R"] + METRIC_NAMES + ["time_s"])
        for name in models:
            for R in sorted(results[name], key=_r_sort_key):
                writer.writerow(
                    [name, R]
                    + [f"{results[name][R][m]:.6f}" for m in METRIC_NAMES]
                    + [f"{times[name][R]:.4f}"]
                )


def _default_summary_csv(output_csv):
    path = Path(output_csv)
    return str(path.with_name(f"{path.stem}_seed_summary{path.suffix}"))


def _write_summary_csv(path, summary):
    columns = ["model", "R", "n_seeds"]
    for m in METRIC_NAMES:
        columns += [f"{m}_mean", f"{m}_std"]
    columns += ["time_s_mean", "time_s_std"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for base, per_r in summary.items():
            for R in sorted(per_r, key=_r_sort_key):
                cell = per_r[R]
                row = [base, R, cell["n"]]
                for m in METRIC_NAMES:
                    row += [f"{cell['mean'][m]:.6f}", f"{cell['std'][m]:.6f}"]
                row += [f"{cell['time_mean']:.4f}", f"{cell['time_std']:.4f}"]
                writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate reconstruction models across acceleration factors."
    )
    parser.add_argument(
        "--root_dir", required=True, help="Preprocessed TrainSet directory."
    )
    parser.add_argument("--split_yaml", default="data/dataset_splits.yaml")
    parser.add_argument("--mode", default="test", choices=["val", "test"])
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--data_format", default="zarr", choices=["zarr", "h5", "mat"])
    parser.add_argument(
        "--accelerations", type=int, nargs="+", default=[10, 20, 30, 40, 50]
    )
    parser.add_argument("--device", default=None)

    parser.add_argument(
        "--no_baselines",
        action="store_true",
        help="Skip the zero-filled and CG-SENSE baselines.",
    )
    parser.add_argument("--cg_iters", type=int, default=20)

    for name, config in TRAINED_MODELS.items():
        key = name.lower()
        parser.add_argument(f"--{key}_ckpt", default=None, help=f"{name} checkpoint.")
        parser.add_argument(
            f"--{key}_config", default=config, help=f"{name} model config."
        )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate exactly these seeds of the given checkpoints.",
    )
    parser.add_argument(
        "--no_seed_expansion",
        action="store_true",
        help="Evaluate only the named checkpoint files, not their sibling seeds.",
    )
    parser.add_argument("--output_csv", default="results.csv")
    parser.add_argument(
        "--summary_csv",
        default=None,
        help="Seed mean/std summary; defaults to <output_csv>_seed_summary.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

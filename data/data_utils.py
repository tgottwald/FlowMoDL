import csv
import os
import random

import h5py
import numcodecs
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False):
    """Seed Python's `random`, NumPy, and PyTorch (CPU + all CUDA devices).

    Call this once at the start of a training run. Note that this does NOT
    seed Numba's internal RNG used inside @njit-compiled functions (e.g.
    data.ktgaussian.fun_mask_gen_2d_numba) -- Numba keeps a separate,
    per-thread RNG state that `np.random.seed()` from ordinary Python code
    has no effect on. That path is seeded per-sample by
    CMRx4DFlowDataset itself (see ktgaussian._seed_numba_rng) when
    the dataset is constructed with `seed=...`.

    Args:
        seed: base seed to apply to all RNGs.
        deterministic: if True, also force cuDNN into deterministic mode
            (disables `cudnn.benchmark` autotuning). This can noticeably
            slow down training and is off by default -- turn it on only
            when you need bitwise-reproducible runs, not just reproducible
            data loading.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def seed_worker(worker_id):
    """`worker_init_fn` for torch.utils.data.DataLoader: re-seeds NumPy and
    Python's `random` from the per-worker torch seed (PyTorch does not do this
    itself) and limits every worker to one thread."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.set_num_threads(1)
    numcodecs.blosc.set_nthreads(1)


def resolve_array_seed(base_seed):
    """
    Adjust `base_seed` for a SLURM job array so each array task gets a
    distinct, deterministic seed: the first task in the array (lowest
    SLURM_ARRAY_TASK_ID) gets exactly `base_seed`, the second gets
    `base_seed + 1`, the third `base_seed + 2`, and so on -- regardless of
    what numeric range (or step) the array was submitted with. E.g.
    `--array=0-9`, `--array=5-14`, and `--array=0-18:2` all make their first
    task use `base_seed` and their second task use `base_seed + 1`.

    Outside a SLURM array (SLURM_ARRAY_TASK_ID unset -- running locally, or
    via `sbatch` without `--array`), this is a no-op: `base_seed` is
    returned unchanged.

    Args:
        base_seed: the configured seed (e.g. cfg.seed), or None if seeding
            is disabled. If None under a SLURM array, seeding stays
            disabled -- there's nothing to offset from.

    Returns:
        The per-task seed (int), or None if seeding is disabled.
    """
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_task_id is None:
        return base_seed

    if base_seed is None:
        print(
            "WARNING: SLURM_ARRAY_TASK_ID is set but the configured `seed` "
            "is null -- seeding stays disabled. Set a base `seed` in the "
            "config to get a distinct per-task seed under a job array."
        )
        return None

    array_task_min = os.environ.get("SLURM_ARRAY_TASK_MIN")
    if array_task_min is None:
        print(
            "WARNING: SLURM_ARRAY_TASK_ID is set but SLURM_ARRAY_TASK_MIN "
            "is not (unusual for a real SLURM array) -- assuming this task "
            "is the first in the array (offset 0)."
        )
        offset = 0
    else:
        # SLURM_ARRAY_TASK_STEP lets --array=MIN-MAX:STEP skip task IDs
        # (e.g. --array=0-18:2 -> task IDs 0, 2, 4, ...). Dividing by it
        # converts the raw ID difference into the task's ordinal position
        # in the array (0-based), which is what should increment the seed
        # by exactly 1 per task regardless of step size.
        array_task_step = os.environ.get("SLURM_ARRAY_TASK_STEP", "1") or "1"
        step = max(int(array_task_step), 1)
        offset = (int(array_task_id) - int(array_task_min)) // step

    seed = base_seed + offset
    print(
        f"SLURM_ARRAY_TASK_ID={array_task_id} (offset {offset} from the "
        f"array's first task) -> seed={seed}"
    )
    return seed


def load_csv_params(filepath):
    """Parse a case's params.csv (header row + value row) into a dict with
    lower-case keys; ';'-separated values become lists."""
    params = {}
    with open(filepath, mode="r") as infile:
        reader = csv.reader(infile)
        rows = list(reader)
        if not rows or len(rows) < 2:
            return params

        def parse_val(val):
            val = val.strip()
            if not val:
                return np.nan

            if ";" in val:
                parts = val.split(";")
                try:
                    return [float(p) for p in parts]
                except ValueError:
                    return [p.strip() for p in parts]
            try:
                return float(val)
            except ValueError:
                return val

        for key, val in zip(rows[0], rows[1]):
            clean_key = key.strip().lower()
            parsed_val = parse_val(val)

            if clean_key == "venc":
                parsed_val = format_venc(parsed_val)

            params[clean_key] = parsed_val

    return params


def derive_compressed_root(data_root):
    """Map a path somewhere under .../TaskR1R2/... to the parallel path
    under .../TaskR1R2_compressed/..., preserving everything else (e.g. the
    trailing /TrainSet or /ValidationSet).

    Used by offline_preprocessing.py and create_test_val_umasks.py to decide
    where to write their (much smaller) zarr-only outputs by default, instead
    of in place next to the raw .mat files. Only the exact "TaskR1R2" path
    segment is replaced (not the substring), so calling this on an
    already-compressed path is a no-op-safe: it raises rather than silently
    double-suffixing.

    Raises:
        ValueError: if no "TaskR1R2" path segment is found in data_root.
    """
    parts = list(os.path.normpath(os.path.abspath(data_root)).split(os.sep))
    try:
        idx = parts.index("TaskR1R2")
    except ValueError:
        raise ValueError(
            f"Could not find a 'TaskR1R2' path segment in '{data_root}' to "
            "derive the parallel TaskR1R2_compressed output location -- pass "
            "--output_root explicitly instead."
        )
    parts[idx] = "TaskR1R2_compressed"
    return os.sep.join(parts)


def format_venc(v_enc_raw):
    """VENC as a list of three floats [v_z, v_y, v_x] (a single value is used for
    all three axes, 150 cm/s if missing)."""
    if isinstance(v_enc_raw, str):
        v_enc_raw = [float(v) for v in v_enc_raw.split(";")]
    elif isinstance(v_enc_raw, (int, float)):
        v_enc_raw = [float(v_enc_raw)]
    if v_enc_raw is None or len(v_enc_raw) == 0:
        return [150.0, 150.0, 150.0]
    if len(v_enc_raw) < 3:
        return [float(v_enc_raw[0])] * 3
    return [float(v) for v in v_enc_raw[:3]]


def h5_first_key(f):
    """First data key of a MATLAB v7.3 (HDF5) file, skipping '#refs#' entries."""
    keys = [k for k in f.keys() if not k.startswith("#")]
    if not keys:
        raise KeyError(f"No dataset found in {f.filename}")
    return keys[0]


def mat_to_tensor(h5_dataset, slices=(), as_complex=False):
    """Read ``h5_dataset[slices]`` of a MATLAB v7.3 file as a tensor. Complex
    MATLAB arrays (stored as a real/imag compound) become complex64; real arrays
    are cast to complex64 only if ``as_complex``."""
    data = h5_dataset[slices]
    if data.dtype.names and "real" in data.dtype.names and "imag" in data.dtype.names:
        return torch.complex(
            torch.from_numpy(np.ascontiguousarray(data["real"])),
            torch.from_numpy(np.ascontiguousarray(data["imag"])),
        ).to(torch.complex64)
    tensor = torch.from_numpy(np.asarray(data))
    return tensor.to(torch.complex64) if as_complex else tensor


def load_mat(path, as_complex=False):
    """Load the (first) array of a MATLAB v7.3 .mat file as a tensor."""
    with h5py.File(path, "r") as f:
        return mat_to_tensor(f[h5_first_key(f)], as_complex=as_complex)

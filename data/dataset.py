"""
Training / validation / test dataset on the CMRx4DFlow training data.

Every item is one case, as a dictionary of unbatched tensors in the axis order
(Nv, Nc, Nt, SPE, PE, FE) = (Nv, Nc, Nt, Z, Y, X):

    kspace_us           (Nv, Nc, Nt, Z, Y, X) complex64  undersampled, normalised
                                                          centred 3D k-space
    target_image        (Nv, Nt, Z, Y, X)     complex64  fully sampled, coil-combined,
                                                          normalised ground truth
    undersampling_mask  (1, 1, Nt, Z, Y, 1)   bool
    sensitivity_maps    (Nc, Z, Y, X)         complex64
    segmask             (Z, Y, X)             bool       aorta ROI
    v_enc               (3,)                  float32    [v_z, v_y, v_x] in cm/s
    usrate_true         scalar                float32    1 / mask density

With ``reconstruct_target_on_gpu=True`` the item instead holds the unmasked,
unnormalised k-space as ``kdata_full``; the training loop then builds
``kspace_us`` and ``target_image`` on the GPU with
``utils.transformations.reconstruct_target_and_normalize``.

In ``train`` mode every item is a random crop (``temporal_crop_size`` cardiac
frames and ``spatial_crop_size[2]`` readout samples; SPE and PE are never
cropped) undersampled with a freshly drawn kt-Gaussian mask at an acceleration
factor drawn from ``accelerations``. In ``val`` / ``test`` mode the full volume
is used together with the pre-generated mask for ``val_acceleration``
(see data/create_test_val_umasks.py).
"""

import os

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
import zarr
from torch.utils.data import Dataset

from data.data_utils import (
    format_venc,
    h5_first_key,
    load_csv_params,
    mat_to_tensor,
)
from data.ktgaussian import generate_ktgaussian_mask
from utils.sense import centered_fftn
from utils.transformations import reconstruct_target_and_normalize


def _collect_all_cases(splits):
    """Every case of the split manifest (test set and every fold's train/val),
    deduplicated on the case path."""
    groups = [splits.get("test_set", [])]
    for fold_data in splits.get("folds", {}).values():
        groups.extend(fold_data[key] for key in ("train", "val") if key in fold_data)

    seen = set()
    cases = []
    for group in groups:
        for case in group:
            key = case.get("path", case.get("id"))
            if key not in seen:
                seen.add(key)
                cases.append(case)
    return cases


def _hybrid_to_centered_kspace(data_x):
    """Hybrid kz-ky-x layout of the preprocessed files -> centred 3D k-space,
    by an FFT along the fully sampled readout axis."""
    return centered_fftn(data_x, dim=(-1,))


class CMRx4DFlowDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split_yaml_path,
        mode="train",
        synthetic=True,
        fold="fold_0",
        ignore_split=False,
        target_nc=None,
        spatial_crop_size=(64, 64, 16),
        spatial_edge_crop_prob=0.0,
        temporal_crop_size=6,
        assume_temporal_cycle=False,
        accelerations=(10, 20, 30, 40, 50),
        val_acceleration=10,
        data_format="zarr",
        seed=None,
        reconstruct_target_on_gpu=False,
        cache_in_memory=False,
    ):
        """
        Args:
            root_dir: directory the case paths of the split manifest are relative to
                (the preprocessed ``.../TrainSet`` for zarr/h5, the raw one for mat).
            split_yaml_path: split manifest (data/dataset_splits.yaml).
            mode: "train", "val" or "test".
            synthetic: mat format only; undersample kdata_full.mat with the
                generated masks instead of reading kdata_ktGaussian{R}.mat.
            fold: fold of the manifest to use for train/val.
            ignore_split: use every case of the manifest (see train.py,
                data.train_on_all_data).
            target_nc: read the coil-compressed files written by
                offline_preprocessing.py --compress_coils --target_nc N.
            spatial_crop_size: (z, y, x) crop size; only x (readout) is used.
            spatial_edge_crop_prob: probability of a readout crop at the FOV edge.
            temporal_crop_size: number of cardiac frames per training crop.
            assume_temporal_cycle: wrap temporal crops around the cardiac cycle.
            accelerations: acceleration factors drawn from in training.
            val_acceleration: acceleration factor in val/test mode.
            data_format: "zarr", "h5" (preprocessed) or "mat" (raw challenge data).
            seed: makes crops and masks reproducible (combined with the epoch,
                see ``set_epoch``).
            reconstruct_target_on_gpu: return ``kdata_full`` instead of
                ``kspace_us`` / ``target_image`` (see module docstring).
            cache_in_memory: cache items (val/test only).
        """
        if mode not in ("train", "val", "test"):
            raise ValueError("mode must be 'train', 'val' or 'test'.")
        self.data_format = data_format.lower()
        if self.data_format not in ("h5", "zarr", "mat"):
            raise ValueError("data_format must be 'h5', 'zarr' or 'mat'.")
        if cache_in_memory and mode == "train":
            raise ValueError(
                "cache_in_memory=True is only supported for mode='val'/'test': "
                "training items are randomly cropped and masked on every access."
            )

        self.root_dir = root_dir
        self.mode = mode
        self.synthetic = synthetic
        self.target_nc = target_nc
        self.spatial_crop_size = spatial_crop_size
        self.edge_crop_prob = spatial_edge_crop_prob
        self.temporal_crop_size = temporal_crop_size
        self.assume_temporal_cycle = assume_temporal_cycle
        self.accelerations = accelerations
        self.val_acceleration = val_acceleration
        self.seed = seed
        self.reconstruct_target_on_gpu = reconstruct_target_on_gpu
        self.cache_in_memory = cache_in_memory
        self._item_cache = {}
        self._mask_cache = {}
        self._file_handles = {}
        # Shared memory, so set_epoch reaches already-forked persistent workers.
        self._epoch = mp.Value("l", 0)

        with open(split_yaml_path, "r") as f:
            splits = yaml.safe_load(f)
        if ignore_split:
            cases = _collect_all_cases(splits)
        elif mode == "test":
            cases = splits["test_set"]
        else:
            cases = splits["folds"][fold][mode]

        self.case_dirs = []
        self.case_files = []
        for case in cases:
            case_dir = os.path.join(self.root_dir, case["path"])
            if self.data_format == "mat":
                file_path = case_dir
            else:
                suffix = f"_nc{self.target_nc}" if self.target_nc is not None else ""
                file_path = os.path.join(
                    case_dir, f"data_compressed{suffix}.{self.data_format}"
                )
            if os.path.exists(file_path):
                self.case_dirs.append(case_dir)
                self.case_files.append(file_path)
            else:
                print(f"Warning: data missing for case {case['id']} at {file_path}")

    def __len__(self):
        return len(self.case_files)

    def set_epoch(self, epoch):
        """Call once per epoch before iterating the DataLoader. The per-item
        random state is derived from (seed, epoch, index), so a fixed seed still
        gives different crops and masks in every epoch."""
        self._epoch.value = epoch

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def _get_handles(self, file_path, case_dir):
        """File handles, cached per worker process."""
        if file_path not in self._file_handles:
            if self.data_format == "mat":
                if self.mode == "train" or self.synthetic:
                    kdata_name = "kdata_full.mat"
                else:
                    kdata_name = f"kdata_ktGaussian{self.val_acceleration}.mat"
                self._file_handles[file_path] = {
                    name: h5py.File(os.path.join(case_dir, fname), "r")
                    for name, fname in (
                        ("kdata", kdata_name),
                        ("smaps", "coilmap.mat"),
                        ("segmask", "segmask.mat"),
                    )
                }
            elif self.data_format == "h5":
                self._file_handles[file_path] = h5py.File(
                    file_path, "r", rdcc_nbytes=1024**2 * 16
                )
            else:
                self._file_handles[file_path] = zarr.open(file_path, mode="r")
        return self._file_handles[file_path]

    def _open_case(self, file_path, case_dir):
        """(kdata, smaps, segmask, v_enc, (Nv, Nc, Nt, Z, Y, X)) for one case."""
        handles = self._get_handles(file_path, case_dir)
        if self.data_format == "mat":
            kdata, smaps, segmask = (
                handles[name][h5_first_key(handles[name])]
                for name in ("kdata", "smaps", "segmask")
            )
            params = load_csv_params(os.path.join(case_dir, "params.csv"))
            v_enc = format_venc(params.get("venc"))
            Nv, Nt, Nc, Z, Y, X = kdata.shape
        else:
            kdata, smaps, segmask = handles["data_x"], handles["smaps"], handles["segmask"]
            v_enc = format_venc(handles.attrs.get("venc"))
            Nv, Nc, Nt, Z, Y, X = kdata.shape
        return kdata, smaps, segmask, v_enc, (Nv, Nc, Nt, Z, Y, X)

    def _read_kspace(self, kdata, t0, t1, xs):
        """Centred 3D k-space of frames [t0, t1) and readout samples ``xs`` as
        (Nv, Nc, T, Z, Y, x)."""
        if self.data_format == "mat":
            chunk = mat_to_tensor(kdata, np.s_[:, t0:t1, :, :, :, xs], as_complex=True)
            return chunk.permute(0, 2, 1, 3, 4, 5)
        return _hybrid_to_centered_kspace(torch.from_numpy(kdata[:, :, t0:t1, :, :, xs]))

    def _load_eval_mask(self, case_dir):
        """Pre-generated mask of ``val_acceleration`` as (1, 1, Nt, Z, Y, 1) bool."""
        R = self.val_acceleration
        key = (case_dir, R)
        if key not in self._mask_cache:
            stem = f"usmask_ktGaussian{R}"
            if self.data_format == "zarr":
                root = zarr.open(os.path.join(case_dir, f"{stem}.zarr"), mode="r")
                mask_np = root[stem if stem in root else list(root.array_keys())[0]][:]
            else:
                ext = "mat" if self.data_format == "mat" else "h5"
                with h5py.File(os.path.join(case_dir, f"{stem}.{ext}"), "r") as f:
                    mask_np = f[h5_first_key(f)][:]
            self._mask_cache[key] = mask_np
        # Stored as (1, Nt, 1, Z, Y, 1), the challenge's layout.
        return torch.from_numpy(self._mask_cache[key]).permute(0, 2, 1, 3, 4, 5).bool()

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------
    def _sample_crop(self, rng, Nt, X):
        """(t_start, t_end, x_start, cx) of the crop."""
        if self.mode != "train":
            return 0, Nt, 0, X

        cx = self.spatial_crop_size[2]
        if rng.random() < self.edge_crop_prob:
            x_start = 0 if rng.random() < 0.5 else max(0, X - cx)
        else:
            x_start = rng.integers(0, max(1, X - cx + 1))

        crop_t = self.temporal_crop_size
        if self.assume_temporal_cycle:
            t_start = rng.integers(0, Nt)
        elif Nt >= crop_t:
            t_start = rng.integers(0, Nt - crop_t + 1)
        else:
            t_start = 0  # padded by repeating the last frame
        return t_start, t_start + crop_t, x_start, cx

    def _read_temporal_crop(self, kdata, Nt, t_start, t_end, xs):
        if t_end <= Nt:
            return self._read_kspace(kdata, t_start, t_end, xs)
        if self.assume_temporal_cycle:
            # Wrap around the end of the cardiac cycle.
            return torch.cat(
                (
                    self._read_kspace(kdata, t_start, Nt, xs),
                    self._read_kspace(kdata, 0, t_end - Nt, xs),
                ),
                dim=2,
            )
        # Short sequence without cycle: repeat the last frame.
        base = self._read_kspace(kdata, t_start, Nt, xs)
        padding = base[:, :, -1:].expand(-1, -1, t_end - Nt, -1, -1, -1)
        return torch.cat((base, padding), dim=2)

    def __getitem__(self, idx):
        if self.cache_in_memory and idx in self._item_cache:
            return {
                k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in self._item_cache[idx].items()
            }

        file_path = self.case_files[idx]
        case_dir = self.case_dirs[idx]

        if self.seed is not None:
            seed_seq = np.random.SeedSequence([self.seed, self._epoch.value, idx])
            rng = np.random.default_rng(seed_seq)
            mask_seed = int(seed_seq.generate_state(1)[0])
        else:
            rng = np.random.default_rng()
            mask_seed = None

        try:
            kdata, smaps, segmask, v_enc, (Nv, Nc, Nt, Z, Y, X) = self._open_case(
                file_path, case_dir
            )
            t_start, t_end, x_start, cx = self._sample_crop(rng, Nt, X)
            xs = slice(x_start, x_start + cx)
            kdata_crop = self._read_temporal_crop(kdata, Nt, t_start, t_end, xs)
            if self.data_format == "mat":
                smaps_crop = mat_to_tensor(smaps, np.s_[:, :, :, xs], as_complex=True)
                segmask_crop = mat_to_tensor(segmask, np.s_[:, :, xs]).bool()
            else:
                smaps_crop = torch.from_numpy(smaps[:, :, :, xs])
                segmask_crop = torch.from_numpy(segmask[:, :, xs])
        except Exception as e:
            raise RuntimeError(f"Error reading {case_dir}: {e}") from e

        if self.mode == "train":
            R = rng.choice(self.accelerations)
            mask_np = generate_ktgaussian_mask(
                Z, Y, kdata_crop.shape[2], R, seed=mask_seed
            )
            mask = torch.from_numpy(mask_np)[None, None, ..., None].bool()
        else:
            mask = self._load_eval_mask(case_dir)
        usrate_true = 1.0 / mask.to(torch.float32).mean()

        result = {
            "undersampling_mask": mask,
            "sensitivity_maps": smaps_crop,
            "segmask": segmask_crop,
            "v_enc": torch.tensor(v_enc, dtype=torch.float32),
            "usrate_true": usrate_true,
        }
        if self.reconstruct_target_on_gpu:
            result["kdata_full"] = kdata_crop
        else:
            kspace_us, target_image, _ = reconstruct_target_and_normalize(
                kdata_crop[None], mask[None], smaps_crop[None], temporal_chunk_size=1
            )
            result["kspace_us"] = kspace_us[0]
            result["target_image"] = target_image[0]

        if self.cache_in_memory:
            self._item_cache[idx] = {
                k: (v.clone() if torch.is_tensor(v) else v) for k, v in result.items()
            }
        return result

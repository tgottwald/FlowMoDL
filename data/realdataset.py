"""
Inference dataset for data in the CMRx4DFlow challenge layout without ground
truth (e.g. the official ValidationSet / TestSet).

Unlike ``data.dataset.CMRx4DFlowDataset`` it needs no split manifest: case
directories are discovered by recursively scanning the given roots, and every
kdata_ktGaussian{R}.mat / usmask_ktGaussian{R}.mat pair present in a case is
one item. Expected per-case files:

    coilmap.mat                (Nc, SPE, PE, FE) complex
    segmask.mat                (SPE, PE, FE) bool
    params.csv                 VENC etc.
    kdata_ktGaussian{R}.mat    (Nv, Nt, Nc, SPE, PE, FE) complex
    usmask_ktGaussian{R}.mat   (1, Nt, 1, SPE, PE, 1) bool

Items use the tensor layout of CMRx4DFlowDataset (kspace_us, undersampling_mask,
sensitivity_maps, segmask, v_enc, usrate_true) plus ``norm``, the normalisation
factor that has to be multiplied back onto the prediction, and bookkeeping
fields (``R``, ``case_dir``, ``out_rel_path``).
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.data_utils import format_venc, load_csv_params, load_mat
from utils.transformations import kspace_norm_factor

STATIC_REQUIRED_FILES = ("coilmap.mat", "segmask.mat", "params.csv")


def _find_available_accelerations(case_dir):
    """Return the sorted list of R values for which both
    kdata_ktGaussian{R}.mat and usmask_ktGaussian{R}.mat exist in case_dir."""
    case_dir = Path(case_dir)
    found = []
    for kfile in case_dir.glob("kdata_ktGaussian*.mat"):
        stem = kfile.name[len("kdata_ktGaussian") : -len(".mat")]
        try:
            r = int(stem)
        except ValueError:
            continue
        if (case_dir / f"usmask_ktGaussian{r}.mat").is_file():
            found.append(r)
    return sorted(found)


def find_real_cases(roots, required_files=STATIC_REQUIRED_FILES):
    """Recursively discover case directories under `roots`.

    A directory qualifies as a case if it contains all of `required_files`
    plus at least one matched (kdata_ktGaussian{R}.mat, usmask_ktGaussian{R}
    .mat) pair. No manifest is used, so this works on directory trees whose
    case IDs are unknown ahead of time.

    Returns
    -------
    list[tuple[str, list[int]]]
        Sorted (case_dir, available_R_list) pairs.
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]

    cases = []
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        # segmask.mat is present for every case (train or test), so anchor on it.
        for anchor in sorted(root.rglob("segmask.mat")):
            case_dir = anchor.parent
            key = str(case_dir)
            if key in seen:
                continue
            if not all((case_dir / f).is_file() for f in required_files):
                continue
            accs = _find_available_accelerations(case_dir)
            if not accs:
                continue
            seen.add(key)
            cases.append((key, accs))

    cases.sort(key=lambda c: c[0])
    return cases


class RealChallengeDataset(Dataset):
    """One item per (case, R) pair, covering the full volume and all Nv
    velocity encodings."""

    def __init__(self, root_dirs, accelerations=None, in_base_dir=None):
        """
        Args:
            root_dirs: str/Path, or list of them, to recursively scan for
                cases. E.g. a single mounted ValidationSet directory, or the
                whole ChallengeData root to cover multiple tasks/sets at once.
            accelerations: optional iterable[int] restricting which R values
                to run. If None (default), every R found per case is used --
                the real ValidationSet ships exactly one R per case, so this
                normally just means "run whatever's there".
            in_base_dir: base directory used to compute each case's relative
                path, which predict.py mirrors under --output_dir to
                reproduce the required
                {Task}/{ValidationSet|TestSet}/{Anatomy}/{Center}/{Vendor}/PXXX/
                layout. Defaults to the common parent of `root_dirs`.
        """
        self.root_dirs = (
            [root_dirs] if isinstance(root_dirs, (str, Path)) else list(root_dirs)
        )
        self.root_dirs = [str(Path(r).resolve()) for r in self.root_dirs]

        if in_base_dir is not None:
            self.in_base_dir = str(Path(in_base_dir).resolve())
        elif len(self.root_dirs) == 1:
            self.in_base_dir = self.root_dirs[0]
        else:
            self.in_base_dir = str(Path(os.path.commonpath(self.root_dirs)))

        accel_filter = set(int(r) for r in accelerations) if accelerations else None

        cases = find_real_cases(self.root_dirs)
        if not cases:
            raise FileNotFoundError(
                f"No valid cases found under {self.root_dirs} -- expected "
                f"{STATIC_REQUIRED_FILES} plus at least one matched "
                "kdata_ktGaussian{R}.mat/usmask_ktGaussian{R}.mat pair per "
                "case directory."
            )

        self.items = []  # (case_dir, R)
        for case_dir, accs in cases:
            use_accs = [r for r in accs if (accel_filter is None or r in accel_filter)]
            for r in use_accs:
                self.items.append((case_dir, r))

        if not self.items:
            raise FileNotFoundError(
                f"Found {len(cases)} case(s) under {self.root_dirs}, but none "
                f"matched the requested accelerations={accelerations}."
            )

        self.items.sort(key=lambda x: (x[0], x[1]))

    def __len__(self):
        return len(self.items)

    def relative_case_path(self, case_dir):
        """Path of `case_dir` relative to in_base_dir -- predict.py joins
        this onto --output_dir to mirror the required submission layout."""
        return os.path.relpath(str(Path(case_dir).resolve()), self.in_base_dir)

    def __getitem__(self, idx):
        case_dir, R = self.items[idx]
        case_dir = Path(case_dir)

        # (Nv, Nt, Nc, SPE, PE, FE) -> (Nv, Nc, Nt, SPE, PE, FE)
        kdata = load_mat(case_dir / f"kdata_ktGaussian{R}.mat", as_complex=True)
        kdata = kdata.permute(0, 2, 1, 3, 4, 5).contiguous()
        smaps = load_mat(case_dir / "coilmap.mat", as_complex=True)
        segmask = load_mat(case_dir / "segmask.mat").bool()
        # (1, Nt, 1, SPE, PE, 1) -> (1, 1, Nt, SPE, PE, 1)
        mask = load_mat(case_dir / f"usmask_ktGaussian{R}.mat")
        mask = mask.permute(0, 2, 1, 3, 4, 5).bool()

        params = load_csv_params(str(case_dir / "params.csv"))
        v_enc = torch.tensor(format_venc(params.get("venc")), dtype=torch.float32)

        # Zero everything outside the mask, then normalise exactly like the
        # training data (utils.transformations.kspace_norm_factor).
        kdata = kdata * mask
        norm_factor = kspace_norm_factor(kdata, mask)
        kspace_us = kdata / norm_factor
        usrate_true = 1.0 / mask.to(torch.float32).mean()

        return {
            "kspace_us": kspace_us,
            "undersampling_mask": mask,
            "sensitivity_maps": smaps,
            "segmask": segmask,
            "v_enc": v_enc,
            "norm": norm_factor,
            "usrate_true": usrate_true,
            "R": int(R),
            "case_dir": str(case_dir),
            "out_rel_path": self.relative_case_path(str(case_dir)),
        }

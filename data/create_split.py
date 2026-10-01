"""
Generate a random train/val/test split of the CMRx4DFlow training cases.

All cases of all centres and vendors are pooled and assigned by one seeded
shuffle (70 % / 15 % / 15 %). The split used in the paper is shipped as
data/dataset_splits.yaml; this script is only needed to create a new one.

Run with:  uv run python -m data.create_split --data_root <.../TaskR1R2/TrainSet>
"""

import argparse
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def parse_dataset(anatomy_dir):
    """All <Center>/<Vendor>/<Patient> case directories below ``anatomy_dir``
    (e.g. .../TrainSet/Aorta), with paths relative to its parent."""
    base_anchor = anatomy_dir.parent
    cases = []
    for center_dir in sorted(p for p in anatomy_dir.iterdir() if p.is_dir()):
        for vendor_dir in sorted(p for p in center_dir.iterdir() if p.is_dir()):
            for patient_dir in sorted(p for p in vendor_dir.iterdir() if p.is_dir()):
                cases.append(
                    {
                        "id": patient_dir.name,
                        "path": patient_dir.relative_to(base_anchor).as_posix(),
                        "center": center_dir.name,
                        "vendor": vendor_dir.name,
                    }
                )
    return cases


def compute_subset_stats(cases_list):
    """Number of cases per centre and scanner."""
    stats = {}
    for case in cases_list:
        entry = stats.setdefault(
            case["center"], {"total_samples": 0, "scanners": Counter()}
        )
        entry["scanners"][case["vendor"]] += 1
        entry["total_samples"] += 1
    return {
        center: {"total_samples": d["total_samples"], "scanners": dict(d["scanners"])}
        for center, d in stats.items()
    }


def generate_random_split(data_root, anatomy, output_yaml, meta_yaml, seed):
    all_cases = parse_dataset(Path(data_root) / anatomy)
    n = len(all_cases)
    if n == 0:
        raise FileNotFoundError(f"No cases found below {Path(data_root) / anatomy}.")
    print(f"Found {n} cases.")

    n_test = round(TEST_FRAC * n)
    n_val = round(VAL_FRAC * n)

    shuffled = list(all_cases)
    random.Random(seed).shuffle(shuffled)
    test_set = shuffled[:n_test]
    val_set = shuffled[n_test : n_test + n_val]
    train_set = shuffled[n_test + n_val :]

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_cases": n,
        "split_type": "random (center/scanner leakage not controlled for)",
    }
    config = {
        "metadata": {
            **metadata,
            "seed": seed,
            "train_frac": TRAIN_FRAC,
            "val_frac": VAL_FRAC,
            "test_frac": TEST_FRAC,
            "note": "Paths relative to the TrainSet directory.",
        },
        "test_set": test_set,
        # One fold, in the {"folds": {name: {"train", "val"}}} layout that
        # data.dataset.CMRx4DFlowDataset reads.
        "folds": {"fold_0": {"train": train_set, "val": val_set}},
    }
    meta_config = {
        "metadata": metadata,
        "train_distribution": compute_subset_stats(train_set),
        "val_distribution": compute_subset_stats(val_set),
        "test_distribution": compute_subset_stats(test_set),
    }

    with open(output_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    with open(meta_yaml, "w") as f:
        yaml.dump(meta_config, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote {output_yaml} and {meta_yaml}.")
    for name, subset in (("Train", train_set), ("Val", val_set), ("Test", test_set)):
        print(f" -> {name}: {len(subset)} cases ({len(subset) / n:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--data_root",
        required=True,
        help="Raw or preprocessed TrainSet directory, e.g. .../TaskR1R2/TrainSet.",
    )
    parser.add_argument("--anatomy", default="Aorta")
    parser.add_argument("--output_yaml", default="data/dataset_splits.yaml")
    parser.add_argument("--meta_yaml", default="data/dataset_meta.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_random_split(
        args.data_root, args.anatomy, args.output_yaml, args.meta_yaml, args.seed
    )

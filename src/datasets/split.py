from pathlib import Path

import json
import random


def split_dataset(
    samples,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    """
    Split medical imaging samples at the patient level.

    The input samples must contain one entry per patient.
    """

    if abs(
        train_ratio + val_ratio + test_ratio - 1.0
    ) > 1e-8:
        raise ValueError(
            "Train/validation/test ratios must sum to 1."
        )

    samples = list(samples)

    rng = random.Random(seed)

    rng.shuffle(samples)

    n = len(samples)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = samples[:n_train]

    val = samples[
        n_train:n_train + n_val
    ]

    test = samples[
        n_train + n_val:
    ]

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def save_split(
    split,
    output_path,
):
    """
    Save patient-level split metadata as JSON.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable = {}

    for split_name, samples in split.items():

        serializable[split_name] = [
            sample["patient_id"]
            for sample in samples
        ]

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            serializable,
            f,
            indent=2,
        )

import json
import gc

import torch
from torch.utils.data import DataLoader

from src.datasets.brats import BraTSDataset
from src.datasets.subset import DatasetSubset
from src.transforms import (
    zscore_normalize,
    normalize_label,
    to_tensor,
)
from src.data.patches import RandomBraTSPatchDataset
from src.utils.seed import set_seed


ROOT = "datasets/raw/Task01_BrainTumour"
SPLIT_PATH = "configs/splits/brats_seed42.json"

PATCH_SIZE = (96, 96, 96)


def transform(image, label):
    image = zscore_normalize(image)
    label = normalize_label(label)
    return to_tensor(image, label)


def main():
    set_seed(42)

    print("=== BUILD BASE DATASET ===")

    base_dataset = BraTSDataset(
        root_dir=ROOT,
        transform=transform,
    )

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    sample_map = {
        sample["patient_id"]: sample
        for sample in base_dataset.samples
    }

    train_ids = split["train"]

    missing = [
        patient_id
        for patient_id in train_ids
        if patient_id not in sample_map
    ]

    if missing:
        raise RuntimeError(
            f"Missing training patients: {missing[:10]}"
        )

    train_samples = [
        sample_map[patient_id]
        for patient_id in train_ids
    ]

    train_subset = DatasetSubset(
        base_dataset,
        train_samples,
    )

    patch_dataset = RandomBraTSPatchDataset(
        dataset=train_subset,
        patch_size=PATCH_SIZE,
        foreground_prob=0.70,
        foreground_classes=(1, 2, 3),
        random_flip=True,
    )

    print("Training patients:", len(train_subset))
    print("Patch dataset length:", len(patch_dataset))

    loader = DataLoader(
        patch_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print("\n=== PATCH EXTRACTION TEST ===")

    checked = 0

    for batch in loader:
        image = batch["image"]
        label = batch["label"]

        patient_id = batch["patient_id"][0]
        sampling_class = int(batch["sampling_class"][0])

        assert image.shape == (1, 4, *PATCH_SIZE)
        assert label.shape == (1, *PATCH_SIZE)

        assert image.dtype == torch.float32
        assert label.dtype == torch.int64

        assert torch.isfinite(image).all()

        observed_labels = set(
            torch.unique(label).tolist()
        )

        assert observed_labels.issubset({0, 1, 2, 3})

        if sampling_class > 0:
            assert sampling_class in observed_labels, (
                f"{patient_id}: selected class "
                f"{sampling_class} absent from extracted patch"
            )

        foreground_fraction = (
            label > 0
        ).float().mean().item()

        print(
            f"{patient_id} | "
            f"class={sampling_class} | "
            f"foreground={foreground_fraction:.4%} | "
            f"shape={tuple(image.shape)}"
        )

        checked += 1

        del batch, image, label
        gc.collect()

        # Initial smoke test: four patients only.
        if checked >= 4:
            break

    assert checked == 4

    print("\n=== RESULT ===")
    print("Patients tested:", checked)
    print("Patch shape:", PATCH_SIZE)
    print("Foreground sampling: PASS")
    print("Label consistency: PASS")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()


import json
from collections import Counter

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

    with open(SPLIT_PATH, encoding="utf-8") as f:
        split = json.load(f)

    base_dataset = BraTSDataset(
        root_dir=ROOT,
        transform=transform,
    )

    sample_map = {
        sample["patient_id"]: sample
        for sample in base_dataset.samples
    }

    train_samples = [
        sample_map[case_id]
        for case_id in split["train"]
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

    loader = DataLoader(
        patch_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    selections = Counter()
    observed_classes = Counter()
    empty_patches = 0
    errors = []

    print("=== FULL TRAINING PATCH AUDIT ===")
    print("Training patients:", len(train_subset))

    for index, batch in enumerate(loader, start=1):
        image = batch["image"]
        label = batch["label"]

        patient_id = batch["patient_id"][0]
        selected_class = int(batch["sampling_class"].item())

        if image.shape != (1, 4, *PATCH_SIZE):
            errors.append((patient_id, "incorrect image shape"))

        if label.shape != (1, *PATCH_SIZE):
            errors.append((patient_id, "incorrect label shape"))

        if not torch.isfinite(image).all():
            errors.append((patient_id, "non-finite image"))

        if image.dtype != torch.float32:
            errors.append((patient_id, "incorrect image dtype"))

        if label.dtype != torch.int64:
            errors.append((patient_id, "incorrect label dtype"))

        unique_labels = set(torch.unique(label).tolist())

        if not unique_labels.issubset({0, 1, 2, 3}):
            errors.append((patient_id, "invalid label values"))

        if selected_class > 0 and selected_class not in unique_labels:
            errors.append((
                patient_id,
                f"selected class {selected_class} missing from patch",
            ))

        selections[selected_class] += 1

        for class_id in unique_labels:
            observed_classes[class_id] += 1

        if not bool((label > 0).any()):
            empty_patches += 1

        if index % 50 == 0 or index == len(train_subset):
            print(f"Processed {index}/{len(train_subset)}")

        del batch, image, label

    foreground_selected = sum(
        count
        for class_id, count in selections.items()
        if class_id > 0
    )

    print("\n=== SAMPLING DISTRIBUTION ===")
    print("Uniform random crops:", selections[-1])
    print("Class 1 crops:", selections[1])
    print("Class 2 crops:", selections[2])
    print("Class 3 crops:", selections[3])

    print(
        "Foreground-centred selections:",
        foreground_selected,
        f"({100 * foreground_selected / len(train_subset):.2f}%)",
    )

    print("\n=== PATCH CONTENT ===")
    print("Patches with no foreground:", empty_patches)
    print("Patches containing class 1:", observed_classes[1])
    print("Patches containing class 2:", observed_classes[2])
    print("Patches containing class 3:", observed_classes[3])

    print("\n=== VALIDATION ===")
    print("Cases processed:", sum(selections.values()))
    print("Errors:", len(errors))

    for error in errors[:20]:
        print(error)

    passed = (
        sum(selections.values()) == len(train_subset)
        and not errors
    )

    print("\nRESULT:", "PASS" if passed else "FAIL")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

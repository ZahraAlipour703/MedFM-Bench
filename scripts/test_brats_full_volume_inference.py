
import json

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.brats import BraTSDataset
from src.transforms import (
    zscore_normalize,
    normalize_label,
    to_tensor,
)
from src.models import build_unet3d
from src.engine.trainer import validate_full_volume
from src.utils.device import get_device
from src.utils.seed import set_seed


ROOT = "datasets/raw/Task01_BrainTumour"
SPLIT_PATH = "configs/splits/brats_seed42.json"


def transform(image, label):
    image = zscore_normalize(image)
    label = normalize_label(label)
    return to_tensor(image, label)


def main():
    set_seed(42)
    torch.set_num_threads(2)

    device = get_device()

    with open(SPLIT_PATH, encoding="utf-8") as f:
        split = json.load(f)

    dataset = BraTSDataset(
        root_dir=ROOT,
        transform=transform,
    )

    sample_map = {
        sample["patient_id"]: index
        for index, sample in enumerate(dataset.samples)
    }

    # Select an ET-positive validation patient.
    patient_id = None

    for candidate in split["val"]:
        label_path = (
            dataset.labels_dir / f"{candidate}.nii.gz"
        )

        label = np.asanyarray(
            nib.load(str(label_path)).dataobj
        )

        if np.any(label == 3):
            patient_id = candidate
            break

    if patient_id is None:
        raise RuntimeError(
            "No ET-positive validation case found."
        )

    sample = dataset[sample_map[patient_id]]

    print("=== REAL FULL-VOLUME INFERENCE TEST ===")
    print("Patient:", patient_id)
    print("Device:", device)
    print("Image shape:", tuple(sample["image"].shape))
    print("Label shape:", tuple(sample["label"].shape))

    assert sample["image"].shape == (4, 240, 240, 155)
    assert sample["label"].shape == (240, 240, 155)
    assert 3 in torch.unique(sample["label"]).tolist()

    loader = DataLoader(
        [sample],
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = build_unet3d(
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    metrics = validate_full_volume(
        model=model,
        loader=loader,
        device=device,
        roi_size=(96, 96, 96),
        overlap=0.25,
        amp_enabled=False,
    )

    print("\n=== INFERENCE RESULTS ===")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    assert metrics["patients"] == 1
    assert metrics["et_positive_patients"] == 1
    assert metrics["et_negative_patients"] == 0

    for key in ("wt_dice", "tc_dice", "et_dice", "score"):
        assert np.isfinite(metrics[key])
        assert 0.0 <= metrics[key] <= 1.0

    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()

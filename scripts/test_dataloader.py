import torch

from src.datasets.brats import BraTSDataset
from src.transforms import (
    zscore_normalize,
    normalize_label,
    to_tensor,
)
from src.data.loader import build_dataloader


def transform(image, label):

    image = zscore_normalize(image)

    label = normalize_label(label)

    image, label = to_tensor(
        image,
        label,
    )

    return image, label


def main():

    dataset = BraTSDataset(
        "datasets/raw/Task01_BrainTumour",
        transform=transform,
    )

    loader = build_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print(
        "Patient:",
        batch["patient_id"][0],
    )

    print(
        "Image:",
        batch["image"].shape,
        batch["image"].dtype,
    )

    print(
        "Label:",
        batch["label"].shape,
        batch["label"].dtype,
    )

    print("=" * 60)

    assert batch["image"].shape[1] == 4

    assert batch["label"].ndim == 4

    assert batch["image"].dtype == torch.float32

    assert batch["label"].dtype == torch.int64

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
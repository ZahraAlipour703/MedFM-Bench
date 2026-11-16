
import json
import torch
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss

from src.datasets.brats import BraTSDataset
from src.datasets.subset import DatasetSubset
from src.transforms import (
    zscore_normalize,
    normalize_label,
    to_tensor,
)
from src.data.patches import RandomBraTSPatchDataset
from src.models import build_unet3d
from src.registry import MODELS
from src.engine.trainer import train_one_epoch
from src.utils.seed import set_seed
from src.utils.device import get_device


ROOT = "datasets/raw/Task01_BrainTumour"
SPLIT_PATH = "configs/splits/brats_seed42.json"
PATCH_SIZE = (32, 32, 32)


def transform(image, label):
    image = zscore_normalize(image)
    label = normalize_label(label)
    return to_tensor(image, label)


def main():
    set_seed(42)
    device = get_device()

    print("=== REAL BRATS TRAINING STEP ===")
    print("Device:", device)

    # Load the validated dataset.
    dataset = BraTSDataset(
        root_dir=ROOT,
        transform=transform,
    )

    with open(SPLIT_PATH, encoding="utf-8") as f:
        split = json.load(f)

    sample_map = {
        sample["patient_id"]: sample
        for sample in dataset.samples
    }

    # Use only the training partition.
    train_samples = [
        sample_map[case_id]
        for case_id in split["train"]
    ]

    train_subset = DatasetSubset(
        dataset,
        train_samples,
    )

    # Extract one real foreground-centred patch.
    patch_dataset = RandomBraTSPatchDataset(
        dataset=train_subset,
        patch_size=PATCH_SIZE,
        foreground_prob=1.0,
        foreground_classes=(1, 2, 3),
        random_flip=True,
    )

    patch = patch_dataset[0]

    loader = DataLoader(
        [patch],
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print("Patient:", patch["patient_id"])
    print("Selected class:", patch["sampling_class"])
    print("Patch image:", tuple(patch["image"].shape))
    print("Patch label:", tuple(patch["label"].shape))
    print("Observed labels:", torch.unique(patch["label"]).tolist())

    assert patch["sampling_class"] in (1, 2, 3)
    assert patch["image"].shape == (4, *PATCH_SIZE)
    assert patch["label"].shape == PATCH_SIZE

    # Build registered model.
    model = MODELS.build(
        "unet3d",
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    criterion = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )

    parameter_before = next(
        model.parameters()
    ).detach().clone()

    # Execute one genuine optimization step.
    result = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scaler=None,
        amp_enabled=False,
        max_grad_norm=12.0,
    )

    parameter_after = next(
        model.parameters()
    ).detach()

    parameters_updated = not torch.equal(
        parameter_before,
        parameter_after,
    )

    assert result["steps"] == 1
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert parameters_updated

    print("\n=== OPTIMIZATION RESULT ===")
    print("Training loss:", result["loss"])
    print("Training steps:", result["steps"])
    print("Parameters updated:", parameters_updated)

    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()

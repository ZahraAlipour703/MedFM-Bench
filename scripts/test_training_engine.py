
import torch
from torch.utils.data import Dataset, DataLoader
from monai.losses import DiceCELoss

from src.models import build_unet3d
from src.engine.trainer import (
    train_one_epoch,
    validate_full_volume,
)


class SyntheticBrainDataset(Dataset):
    def __init__(self, count=2, size=32):
        self.count = count
        self.size = size

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(100 + index)

        image = torch.randn(
            4,
            self.size,
            self.size,
            self.size,
            generator=generator,
        )

        # Patient 0: includes all four classes.
        # Patient 1: ET-negative; labels are only 0, 1, 2.
        max_label = 4 if index == 0 else 3

        label = torch.randint(
            0,
            max_label,
            (self.size, self.size, self.size),
            generator=generator,
            dtype=torch.int64,
        )

        return {
            "patient_id": f"SYNTHETIC_{index:03d}",
            "image": image,
            "label": label,
        }


def main():
    torch.manual_seed(42)

    device = torch.device("cpu")

    dataset = SyntheticBrainDataset(count=2, size=32)

    loader = DataLoader(
        dataset,
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

    print("=== TRAINING ENGINE TEST ===")

    train_result = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scaler=None,
        amp_enabled=False,
        max_grad_norm=12.0,
    )

    print("Training loss:", train_result["loss"])
    print("Training steps:", train_result["steps"])

    assert train_result["steps"] == 2
    assert torch.isfinite(torch.tensor(train_result["loss"]))

    print("\n=== VALIDATION ENGINE TEST ===")

    val_result = validate_full_volume(
        model=model,
        loader=loader,
        device=device,
        roi_size=(32, 32, 32),
        overlap=0.25,
        amp_enabled=False,
    )

    for key, value in val_result.items():
        print(f"{key}: {value}")

    assert val_result["patients"] == 2
    assert 0.0 <= val_result["wt_dice"] <= 1.0
    assert 0.0 <= val_result["tc_dice"] <= 1.0
    assert 0.0 <= val_result["et_dice"] <= 1.0

    assert val_result["et_positive_patients"] == 1
    assert val_result["et_negative_patients"] == 1

    assert val_result["et_negative_fp_voxels_mean"] >= 0
    assert val_result["et_negative_fp_voxels_max"] >= 0

    print("\nET-negative patient handling: PASS")
    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()


import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from monai.losses import DiceCELoss
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.patches import RandomBraTSPatchDataset
from src.datasets.brats import BraTSDataset
from src.datasets.subset import DatasetSubset
from src.engine.trainer import (
    train_one_epoch,
    validate_full_volume,
)
from src.models import build_unet3d
from src.registry import MODELS
from src.transforms import (
    zscore_normalize,
    normalize_label,
    to_tensor,
)
from src.utils.seed import set_seed


def resolve_path(repo_root, value):
    path = Path(value)

    if not path.is_absolute():
        path = repo_root / path

    return path.resolve()


def transform(image, label):
    image = zscore_normalize(image)
    label = normalize_label(label)
    return to_tensor(image, label)


def load_and_validate_data(cfg, repo_root):
    root = resolve_path(
        repo_root,
        cfg.dataset.root_dir,
    )

    split_path = resolve_path(
        repo_root,
        cfg.dataset.split_file,
    )

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}"
        )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}"
        )

    dataset = BraTSDataset(
        root_dir=root,
        transform=transform,
    )

    with split_path.open("r", encoding="utf-8") as f:
        split = json.load(f)

    required = {"train", "val", "test"}

    if not required.issubset(split):
        raise ValueError(
            f"Split file must contain {sorted(required)}"
        )

    dataset_ids = {
        sample["patient_id"]
        for sample in dataset.samples
    }

    partitions = {
        name: list(split[name])
        for name in required
    }

    # Check duplicate patient assignments.
    for name, ids in partitions.items():
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"Duplicate patients in {name} partition."
            )

    train_ids = set(partitions["train"])
    val_ids = set(partitions["val"])
    test_ids = set(partitions["test"])

    if train_ids & val_ids:
        raise ValueError("Training/validation overlap.")

    if train_ids & test_ids:
        raise ValueError("Training/test overlap.")

    if val_ids & test_ids:
        raise ValueError("Validation/test overlap.")

    assigned = train_ids | val_ids | test_ids

    if assigned != dataset_ids:
        raise ValueError(
            "Split does not exactly match the dataset. "
            f"Missing: {len(dataset_ids - assigned)}, "
            f"unknown: {len(assigned - dataset_ids)}"
        )

    sample_map = {
        sample["patient_id"]: sample
        for sample in dataset.samples
    }

    train_samples = [
        sample_map[case_id]
        for case_id in partitions["train"]
    ]

    val_samples = [
        sample_map[case_id]
        for case_id in partitions["val"]
    ]

    train_subset = DatasetSubset(
        dataset,
        train_samples,
    )

    val_subset = DatasetSubset(
        dataset,
        val_samples,
    )

    patch_size = tuple(
        int(x) for x in cfg.training.patch_size
    )

    strides = tuple(
        int(x) for x in cfg.model.strides
    )

    divisor = math.prod(strides)

    if any(
        dimension % divisor != 0
        for dimension in patch_size
    ):
        raise ValueError(
            f"Patch dimensions must be divisible by "
            f"{divisor}; received {patch_size}."
        )

    patch_dataset = RandomBraTSPatchDataset(
        dataset=train_subset,
        patch_size=patch_size,
        foreground_prob=float(
            cfg.training.foreground_prob
        ),
        foreground_classes=(1, 2, 3),
        random_flip=True,
    )

    return (
        dataset,
        split,
        patch_dataset,
        val_subset,
    )


def build_model(cfg, device):
    model = MODELS.build(
        cfg.model.name,
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        channels=tuple(cfg.model.channels),
        strides=tuple(cfg.model.strides),
        num_res_units=int(cfg.model.num_res_units),
        norm=str(cfg.model.norm),
        dropout=float(cfg.model.dropout),
    )

    return model.to(device)


def run_preflight(cfg, repo_root):
    print("\n=== MEDFM-BENCH PREFLIGHT ===")

    set_seed(int(cfg.seed))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    dataset, split, patch_dataset, val_subset = (
        load_and_validate_data(cfg, repo_root)
    )

    print("\n=== PARTITION INTEGRITY ===")
    print("Total cases:", len(dataset))
    print("Training:", len(split["train"]))
    print("Validation:", len(split["val"]))
    print("Internal test:", len(split["test"]))

    print("\nInternal test partition will remain unused.")

    print("\n=== PATCH PREFLIGHT ===")

    patch = patch_dataset[0]

    expected_image_shape = (
        4,
        *tuple(int(x) for x in cfg.training.patch_size),
    )

    expected_label_shape = tuple(
        int(x) for x in cfg.training.patch_size
    )

    assert tuple(patch["image"].shape) == expected_image_shape
    assert tuple(patch["label"].shape) == expected_label_shape

    assert patch["image"].dtype == torch.float32
    assert patch["label"].dtype == torch.int64
    assert torch.isfinite(patch["image"]).all()

    observed_labels = set(
        torch.unique(patch["label"]).tolist()
    )

    assert observed_labels.issubset({0, 1, 2, 3})

    print("Patient:", patch["patient_id"])
    print("Selected class:", patch["sampling_class"])
    print("Image shape:", tuple(patch["image"].shape))
    print("Label shape:", tuple(patch["label"].shape))
    print("Observed labels:", sorted(observed_labels))

    del patch

    print("\n=== MODEL PREFLIGHT ===")

    model = build_model(cfg, device)

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Model:", cfg.model.name)
    print("Parameters:", parameters)
    print("Input channels:", cfg.model.in_channels)
    print("Output classes:", cfg.model.out_channels)

    del model

    print("\n=== PREFLIGHT RESULT ===")
    print("Dataset split: PASS")
    print("Patch extraction: PASS")
    print("Model construction: PASS")
    print("Validation patients:", len(val_subset))
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print(
            "\nCPU preflight completed. "
            "Full training requires a CUDA GPU."
        )

    print("\nPREFLIGHT: PASS")


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_score,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_validation_score": best_score,
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, path)


def run_training(cfg, repo_root):
    device = torch.device("cuda")

    set_seed(int(cfg.seed))

    torch.backends.cudnn.benchmark = False

    (
        dataset,
        split,
        patch_dataset,
        val_subset,
    ) = load_and_validate_data(cfg, repo_root)

    workers = int(cfg.training.num_workers)

    train_loader = DataLoader(
        patch_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = build_model(cfg, device)

    criterion = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    amp_enabled = bool(cfg.training.amp)

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
        if amp_enabled
        else None
    )

    output_root = resolve_path(
        repo_root,
        cfg.experiment.output_dir,
    )

    run_name = datetime.now().strftime(
        "run_%Y%m%d_%H%M%S"
    )

    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    OmegaConf.save(
        cfg,
        run_dir / "config.yaml",
    )

    print("\n=== TRAINING STARTED ===")
    print("Run directory:", run_dir)
    print("Training patients:", len(split["train"]))
    print("Validation patients:", len(split["val"]))
    print("Internal test patients: reserved")
    print("Steps per epoch:", len(train_loader))
    print("Device:", device)
    print("AMP:", amp_enabled)

    history_path = run_dir / "history.csv"

    fields = [
        "epoch",
        "train_loss",
        "validation_score",
        "wt_dice",
        "tc_dice",
        "et_dice",
        "et_negative_patients",
        "et_negative_fp_mean",
        "et_negative_fp_max",
        "learning_rate",
        "epoch_seconds",
    ]

    best_score = float("-inf")
    early_stop_best = float("-inf")
    bad_epochs = 0

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields,
        )

        writer.writeheader()

        for epoch in range(
            1,
            int(cfg.training.epochs) + 1,
        ):
            start_time = time.time()

            train_result = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                amp_enabled=amp_enabled,
                max_grad_norm=float(
                    cfg.training.max_grad_norm
                ),
            )

            validation = validate_full_volume(
                model=model,
                loader=val_loader,
                device=device,
                roi_size=tuple(cfg.validation.roi_size),
                overlap=float(cfg.validation.overlap),
                amp_enabled=amp_enabled,
            )

            score = float(validation["score"])

            scheduler.step(score)

            current_lr = float(
                optimizer.param_groups[0]["lr"]
            )

            elapsed = time.time() - start_time

            print(
                f"\nEpoch {epoch:03d} | "
                f"Train loss {train_result['loss']:.5f} | "
                f"WT {validation['wt_dice']:.4f} | "
                f"TC {validation['tc_dice']:.4f} | "
                f"ET {validation['et_dice']:.4f} | "
                f"Score {score:.4f} | "
                f"LR {current_lr:.2e} | "
                f"{elapsed:.1f}s"
            )

            row = {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "validation_score": score,
                "wt_dice": validation["wt_dice"],
                "tc_dice": validation["tc_dice"],
                "et_dice": validation["et_dice"],
                "et_negative_patients": (
                    validation["et_negative_patients"]
                ),
                "et_negative_fp_mean": (
                    validation["et_negative_fp_voxels_mean"]
                ),
                "et_negative_fp_max": (
                    validation["et_negative_fp_voxels_max"]
                ),
                "learning_rate": current_lr,
                "epoch_seconds": elapsed,
            }

            writer.writerow(row)
            csv_file.flush()

            # Save the highest-scoring checkpoint.
            if score > best_score:
                best_score = score

                save_checkpoint(
                    run_dir / "best_model.pt",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    best_score,
                )

                print("Best checkpoint updated.")

            # Early stopping uses min_delta.
            if score > early_stop_best + float(
                cfg.training.min_delta
            ):
                early_stop_best = score
                bad_epochs = 0
            else:
                bad_epochs += 1

            save_checkpoint(
                run_dir / "latest_checkpoint.pt",
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_score,
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if bad_epochs >= int(cfg.training.patience):
                print(
                    f"\nEarly stopping after {epoch} epochs."
                )
                break

    print("\n=== EXPERIMENT FINISHED ===")
    print("Best validation score:", best_score)
    print("Output directory:", run_dir)
    print("History:", history_path)


def main():
    parser = argparse.ArgumentParser(
        description="MedFM-Bench 3D U-Net experiment runner"
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate data, patch extraction and model without training.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    config_path = resolve_path(
        repo_root,
        args.config,
    )

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    cfg = OmegaConf.load(config_path)

    if args.preflight:
        run_preflight(cfg, repo_root)
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Full 3D training is not "
            "permitted on the CPU-only preflight environment. "
            "Run this experiment on a CUDA-enabled GPU."
        )

    run_training(cfg, repo_root)


if __name__ == "__main__":
    main()

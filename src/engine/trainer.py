
from contextlib import nullcontext

import torch
from monai.inferers import sliding_window_inference


REGIONS = {
    "WT": (1, 2, 3),
    "TC": (2, 3),
    "ET": (3,),
}


def autocast_context(device, enabled=False):
    """Enable FP16 autocast only on CUDA."""
    if enabled and device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    amp_enabled=False,
    max_grad_norm=12.0,
):
    """Train for one epoch using 3D patches."""

    model.train()

    total_loss = 0.0
    steps = 0

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=(device.type == "cuda"),
        )

        labels = batch["label"].unsqueeze(1).to(
            device,
            dtype=torch.long,
            non_blocking=(device.type == "cuda"),
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at step {steps + 1}"
            )

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )

        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_loss += float(loss.detach().item())
        steps += 1

    if steps == 0:
        raise RuntimeError("Training loader produced no batches.")

    return {
        "loss": total_loss / steps,
        "steps": steps,
    }


@torch.inference_mode()
def validate_full_volume(
    model,
    loader,
    device,
    roi_size=(96, 96, 96),
    overlap=0.5,
    amp_enabled=False,
):
    """
    Evaluate full 3D volumes using sliding-window inference.

    WT and TC Dice are averaged over all patients.
    ET Dice is averaged over ET-positive patients only.
    ET-negative patients are evaluated using false-positive
    ET voxel counts.
    """

    model.eval()

    region_scores = {
        name: []
        for name in REGIONS
    }

    et_negative_fp_voxels = []
    et_positive_patients = 0
    et_negative_patients = 0
    patient_count = 0

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=(device.type == "cuda"),
        )

        labels = batch["label"].to(
            device,
            dtype=torch.long,
            non_blocking=(device.type == "cuda"),
        )

        if images.shape[0] != 1:
            raise ValueError(
                "Full-volume validation requires batch_size=1."
            )

        if images.ndim != 5 or labels.ndim != 4:
            raise ValueError("Unexpected validation tensor dimensions.")

        with autocast_context(device, amp_enabled):
            logits = sliding_window_inference(
                inputs=images,
                roi_size=tuple(roi_size),
                sw_batch_size=1,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )

        predictions = logits.argmax(dim=1)

        for pred, target in zip(predictions, labels):
            patient_count += 1

            for region_name, class_ids in REGIONS.items():
                pred_mask = torch.zeros_like(
                    pred,
                    dtype=torch.bool,
                )
                target_mask = torch.zeros_like(
                    target,
                    dtype=torch.bool,
                )

                for class_id in class_ids:
                    pred_mask |= pred == class_id
                    target_mask |= target == class_id

                pred_count = int(pred_mask.sum().item())
                target_count = int(target_mask.sum().item())

                if region_name == "ET" and target_count == 0:
                    et_negative_patients += 1
                    et_negative_fp_voxels.append(pred_count)
                    continue

                if region_name == "ET":
                    et_positive_patients += 1

                intersection = int(
                    (pred_mask & target_mask).sum().item()
                )

                dice = (
                    2.0 * intersection + 1e-5
                ) / (
                    pred_count + target_count + 1e-5
                )

                region_scores[region_name].append(dice)

        del images, labels, logits, predictions

    if patient_count == 0:
        raise RuntimeError("Validation loader produced no patients.")

    if not region_scores["ET"]:
        raise RuntimeError(
            "Validation set contains no ET-positive patients."
        )

    means = {
        name: sum(values) / len(values)
        for name, values in region_scores.items()
    }

    validation_score = sum(means.values()) / 3.0

    return {
        "wt_dice": means["WT"],
        "tc_dice": means["TC"],
        "et_dice": means["ET"],
        "score": validation_score,
        "patients": patient_count,
        "et_positive_patients": et_positive_patients,
        "et_negative_patients": et_negative_patients,
        "et_negative_fp_voxels_mean": (
            sum(et_negative_fp_voxels) /
            len(et_negative_fp_voxels)
            if et_negative_fp_voxels else 0.0
        ),
        "et_negative_fp_voxels_max": (
            max(et_negative_fp_voxels)
            if et_negative_fp_voxels else 0
        ),
    }

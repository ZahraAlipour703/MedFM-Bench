
import torch
from monai.losses import DiceCELoss

from src.models import build_unet3d
from src.registry import MODELS


def main():
    torch.manual_seed(42)

    print("=== MODEL REGISTRY ===")

    print("Registered models:", MODELS.keys())

    assert "unet3d" in MODELS.keys()

    model = MODELS.build(
        "unet3d",
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )

    criterion = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )

    image = torch.randn(
        1, 4, 32, 32, 32
    )

    label = torch.randint(
        0, 4,
        (1, 1, 32, 32, 32),
        dtype=torch.long,
    )

    model.train()

    logits = model(image)
    loss = criterion(logits, label)

    assert logits.shape == (1, 4, 32, 32, 32)
    assert torch.isfinite(loss)

    loss.backward()

    gradients = [
        p.grad
        for p in model.parameters()
        if p.grad is not None
    ]

    assert gradients, "No gradients were produced."

    assert all(
        torch.isfinite(g).all()
        for g in gradients
    ), "Non-finite gradients detected."

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\n=== MODEL TEST ===")
    print("Parameters:", parameters)
    print("Input:", tuple(image.shape))
    print("Output:", tuple(logits.shape))
    print("Loss:", float(loss.detach()))
    print("Gradient tensors:", len(gradients))
    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()

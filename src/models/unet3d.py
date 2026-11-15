
from monai.networks.nets import UNet

from src.registry import MODELS


@MODELS.register("unet3d")
def build_unet3d(
    in_channels=4,
    out_channels=4,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
    norm="INSTANCE",
    dropout=0.0,
):
    """
    Build a 3D U-Net for multimodal brain MRI segmentation.

    Input:
        (B, 4, H, W, D)

    Output:
        (B, 4, H, W, D)
    """

    channels = tuple(int(c) for c in channels)
    strides = tuple(int(s) for s in strides)

    if in_channels <= 0 or out_channels <= 0:
        raise ValueError("Channel counts must be positive.")

    if len(channels) != len(strides) + 1:
        raise ValueError(
            "The number of strides must equal "
            "the number of channel levels minus one."
        )

    if any(c <= 0 for c in channels):
        raise ValueError("All channel widths must be positive.")

    if any(s <= 0 for s in strides):
        raise ValueError("All strides must be positive.")

    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
    )

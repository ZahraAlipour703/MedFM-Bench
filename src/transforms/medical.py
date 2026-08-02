import numpy as np
import torch


def zscore_normalize(
    image: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize each MRI modality independently.

    Parameters
    ----------
    image:
        MRI volume with shape (H, W, D, C).

    eps:
        Numerical stability constant.

    Returns
    -------
    np.ndarray
        Normalized image with the same shape.
    """

    image = image.astype(np.float32, copy=False)

    normalized = np.zeros_like(image)

    for channel in range(image.shape[-1]):

        modality = image[..., channel]

        # MRI background is generally zero.
        foreground = modality != 0

        if not np.any(foreground):
            continue

        values = modality[foreground]

        mean = values.mean()
        std = values.std()

        normalized_channel = np.zeros_like(modality)

        if std > eps:
            normalized_channel[foreground] = (
                modality[foreground] - mean
            ) / std

        normalized[..., channel] = normalized_channel

    return normalized


def normalize_label(label: np.ndarray) -> np.ndarray:
    """
    Convert segmentation labels to integer class IDs.
    """

    label = np.asarray(label)

    return label.astype(np.int64)


def to_tensor(
    image: np.ndarray,
    label: np.ndarray,
):
    """
    Convert NumPy arrays to PyTorch tensors.

    Input:
        image: (H, W, D, C)
        label: (H, W, D)

    Output:
        image: (C, H, W, D)
        label: (H, W, D)
    """

    image = np.transpose(
        image,
        (3, 0, 1, 2),
    )

    image_tensor = torch.from_numpy(
        image.copy()
    ).float()

    label_tensor = torch.from_numpy(
        label.copy()
    ).long()

    return image_tensor, label_tensor
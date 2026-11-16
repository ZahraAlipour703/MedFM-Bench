
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class RandomBraTSPatchDataset(Dataset):
    """
    Extract random 3D patches from full BraTS patient volumes.

    Each dataset item loads one patient and extracts one patch.
    The source dataset is not modified.

    Expected image shape: (4, H, W, D)
    Expected label shape: (H, W, D)
    """

    def __init__(
        self,
        dataset,
        patch_size=(96, 96, 96),
        foreground_prob=0.70,
        foreground_classes=(1, 2, 3),
        random_flip=True,
    ):
        self.dataset = dataset

        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3

        self.patch_size = tuple(int(x) for x in patch_size)

        if len(self.patch_size) != 3:
            raise ValueError("patch_size must contain three dimensions.")

        if any(x <= 0 for x in self.patch_size):
            raise ValueError("Patch dimensions must be positive.")

        if not 0.0 <= foreground_prob <= 1.0:
            raise ValueError("foreground_prob must be between 0 and 1.")

        self.foreground_prob = float(foreground_prob)
        self.foreground_classes = tuple(
            int(x) for x in foreground_classes
        )
        self.random_flip = random_flip

    def __len__(self):
        return len(self.dataset)

    def _get_patch_start(self, center, shape):
        starts = []

        for c, dim, patch in zip(
            center, shape, self.patch_size
        ):
            start = int(c) - patch // 2
            start = max(0, min(start, dim - patch))
            starts.append(start)

        return tuple(starts)

    def __getitem__(self, index):
        sample = self.dataset[index]

        image = sample["image"]
        label = sample["label"]

        if not torch.is_tensor(image):
            raise TypeError("Image must be a PyTorch tensor.")

        if not torch.is_tensor(label):
            raise TypeError("Label must be a PyTorch tensor.")

        if image.dtype != torch.float32:
            raise TypeError("Image must have dtype float32.")

        if label.dtype != torch.int64:
            raise TypeError("Label must have dtype int64.")

        if image.ndim != 4 or image.shape[0] != 4:
            raise ValueError(
                f"Expected image shape (4,H,W,D), got {tuple(image.shape)}"
            )

        if label.ndim != 3:
            raise ValueError("Expected a 3D segmentation label.")

        if tuple(image.shape[1:]) != tuple(label.shape):
            raise ValueError("Image and label shapes do not match.")

        spatial_shape = tuple(int(x) for x in label.shape)

        if any(
            patch > dim
            for patch, dim in zip(self.patch_size, spatial_shape)
        ):
            raise ValueError(
                f"Patch size {self.patch_size} exceeds "
                f"volume size {spatial_shape}."
            )

        # Find available foreground classes for this patient.
        label_np = label.numpy()

        class_coordinates = {
            class_id: np.argwhere(label_np == class_id)
            for class_id in self.foreground_classes
        }

        available_classes = [
            class_id
            for class_id, coordinates in class_coordinates.items()
            if len(coordinates) > 0
        ]

        use_foreground = (
            bool(available_classes)
            and random.random() < self.foreground_prob
        )

        if use_foreground:
            sampling_class = random.choice(available_classes)

            coordinates = class_coordinates[sampling_class]

            selected = coordinates[
                np.random.randint(len(coordinates))
            ]

            start = self._get_patch_start(
                selected,
                spatial_shape,
            )

        else:
            # Uniformly sample valid crop origins.
            sampling_class = -1

            start = tuple(
                random.randint(0, dim - patch)
                for dim, patch in zip(
                    spatial_shape,
                    self.patch_size,
                )
            )

        slices = tuple(
            slice(s, s + p)
            for s, p in zip(start, self.patch_size)
        )

        patch_image = image[(slice(None),) + slices].contiguous()
        patch_label = label[slices].contiguous()

        # Apply identical spatial flips to image and label.
        if self.random_flip:
            for axis in range(3):
                if random.random() < 0.5:
                    patch_image = torch.flip(
                        patch_image,
                        dims=(axis + 1,),
                    )
                    patch_label = torch.flip(
                        patch_label,
                        dims=(axis,),
                    )

        return {
            "patient_id": sample["patient_id"],
            "image": patch_image,
            "label": patch_label,
            "sampling_class": sampling_class,
        }

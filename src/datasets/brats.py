from pathlib import Path

import nibabel as nib
import numpy as np

from .base_dataset import BaseMedicalDataset
from src.registry import DATASETS


@DATASETS.register("brats")
class BraTSDataset(BaseMedicalDataset):
    """
    BraTS dataset loader.

    Supports:
        - dataset discovery
        - patient-level subsets
        - preprocessing
        - PyTorch-compatible indexing
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        samples=None,
    ):
        super().__init__(
            root_dir=root_dir,
            transform=transform,
        )

        self.root_dir = Path(root_dir)

        self.images_dir = self.root_dir / "imagesTr"
        self.labels_dir = self.root_dir / "labelsTr"

        if samples is None:
            self.samples = self.load_data()
        else:
            self.samples = samples

    def load_data(self):
        """
        Discover all image/label pairs.
        """

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}"
            )

        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f"Labels directory not found: {self.labels_dir}"
            )

        samples = []

        image_files = sorted(
            self.images_dir.glob("BRATS_*.nii.gz")
        )

        for image_path in image_files:

            patient_id = image_path.name.replace(
                ".nii.gz",
                "",
            )

            label_path = (
                self.labels_dir
                / f"{patient_id}.nii.gz"
            )

            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing label for {patient_id}"
                )

            samples.append(
                {
                    "patient_id": patient_id,
                    "image": image_path,
                    "label": label_path,
                }
            )

        if not samples:
            raise RuntimeError(
                "No BraTS samples found."
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image = nib.load(
            sample["image"]
        ).get_fdata(
            dtype=np.float32
        )

        label = nib.load(
            sample["label"]
        ).get_fdata(
            dtype=np.float32
        )

        if self.transform is not None:
            image, label = self.transform(
                image,
                label,
            )

        return {
            "patient_id": sample["patient_id"],
            "image": image,
            "label": label,
        }
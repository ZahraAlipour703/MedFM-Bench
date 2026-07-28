from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetScanner:
    """
    Scans a medical dataset directory and creates a standardized
    index of valid image-mask pairs.

    Expected directory structure (generic):

        root_dir/
        ├── Patient001/
        │   ├── image.nii.gz
        │   └── mask.nii.gz
        ├── Patient002/
        │   ├── image.nii.gz
        │   └── mask.nii.gz
        └── ...
    """

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

    def scan(self):
        """
        Scan the dataset directory and return a list of valid samples.

        Returns
        -------
        list[dict]
            Each dictionary contains:
                - patient_id
                - image
                - mask
        """

        logger.info(f"Scanning dataset: {self.root_dir}")

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root_dir}"
            )

        samples = []

        patient_dirs = sorted(
            [p for p in self.root_dir.iterdir() if p.is_dir()]
        )

        logger.info(f"Found {len(patient_dirs)} patient folders.")

        for patient in patient_dirs:

            image = patient / "image.nii.gz"
            mask = patient / "mask.nii.gz"

            image_exists = image.exists()
            mask_exists = mask.exists()

            if not image_exists:
                logger.warning(
                    f"{patient.name}: image.nii.gz not found."
                )

            if not mask_exists:
                logger.warning(
                    f"{patient.name}: mask.nii.gz not found."
                )

            if image_exists and mask_exists:

                samples.append(
                    {
                        "patient_id": patient.name,
                        "image": image,
                        "mask": mask,
                    }
                )

        logger.info(
            f"Dataset scan completed successfully."
        )

        logger.info(
            f"Valid samples: {len(samples)}"
        )

        return samples
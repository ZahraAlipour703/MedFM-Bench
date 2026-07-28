from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DatasetScanner:
    """
    Scans a medical dataset directory and creates
    a standardized index of samples.
    """

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

    def scan(self):
        samples = []
        logger.info("Scanning dataset...")

        for patient in sorted(self.root_dir.iterdir()):

            if not patient.is_dir():
                continue

            image = patient / "image.nii.gz"
            mask = patient / "mask.nii.gz"

            if image.exists() and mask.exists():

                samples.append(
                    {
                        "patient_id": patient.name,
                        "image": image,
                        "mask": mask,
                    }
                )
            logger.info(f"Found {len(samples)} valid patients.")

        return samples
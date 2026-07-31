from pathlib import Path
from typing import Dict

import nibabel as nib

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetValidator:
    """
    Validate the structure and integrity of a medical segmentation dataset.

    Current checks:
        - dataset.json exists
        - imagesTr directory exists
        - labelsTr directory exists
        - every training image has a corresponding label
        - every image/label can be opened
        - image and label have identical shapes
    """

    def __init__(self, dataset_root: str | Path):
        self.root = Path(dataset_root)

        self.images_dir = self.root / "imagesTr"
        self.labels_dir = self.root / "labelsTr"
        self.dataset_json = self.root / "dataset.json"

    def validate(self) -> Dict[str, int | bool]:
        """
        Run the complete validation pipeline.

        Returns
        -------
        dict
            Validation summary.
        """

        logger.info("=" * 60)
        logger.info("Medical Dataset Validation")
        logger.info("=" * 60)

        report = {
            "images": 0,
            "labels": 0,
            "missing_labels": 0,
            "corrupted_files": 0,
            "shape_mismatch": 0,
            "status": False,
        }

        # ---------------------------------------------------
        # Check required files/directories
        # ---------------------------------------------------

        if not self.dataset_json.exists():
            raise FileNotFoundError(
                f"dataset.json not found:\n{self.dataset_json}"
            )

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"imagesTr folder not found:\n{self.images_dir}"
            )

        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f"labelsTr folder not found:\n{self.labels_dir}"
            )

        logger.info("✓ Required files found.")

        image_files = sorted(self.images_dir.glob("*.nii.gz"))

        report["images"] = len(image_files)

        # ---------------------------------------------------
        # Validate every case
        # ---------------------------------------------------

        for image_path in image_files:

            label_path = self.labels_dir / image_path.name

            if not label_path.exists():
                logger.warning(f"Missing label: {label_path.name}")
                report["missing_labels"] += 1
                continue

            report["labels"] += 1

            try:

                image = nib.load(image_path)
                label = nib.load(label_path)

            except Exception as e:
                logger.error(f"Cannot read {image_path.name}: {e}")
                report["corrupted_files"] += 1
                continue

            if image.shape != label.shape:

                logger.warning(
                    f"Shape mismatch: {image_path.name} "
                    f"{image.shape} != {label.shape}"
                )

                report["shape_mismatch"] += 1

        # ---------------------------------------------------
        # Final status
        # ---------------------------------------------------

        report["status"] = (
            report["missing_labels"] == 0
            and report["corrupted_files"] == 0
            and report["shape_mismatch"] == 0
        )

        self._print_report(report)

        return report

    def _print_report(self, report: Dict) -> None:
        """
        Print a validation summary.
        """

        logger.info("")
        logger.info("=" * 60)
        logger.info("Validation Report")
        logger.info("=" * 60)

        logger.info(f"Dataset root      : {self.root}")
        logger.info(f"Training images   : {report['images']}")
        logger.info(f"Training labels   : {report['labels']}")
        logger.info(f"Missing labels    : {report['missing_labels']}")
        logger.info(f"Corrupted files   : {report['corrupted_files']}")
        logger.info(f"Shape mismatches  : {report['shape_mismatch']}")

        status = "PASS" if report["status"] else "FAIL"

        logger.info(f"Dataset Status    : {status}")
        logger.info("=" * 60)
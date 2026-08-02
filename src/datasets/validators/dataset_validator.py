from pathlib import Path
from typing import Dict, List

import nibabel as nib

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetValidator:
    """
    Validator for the Medical Segmentation Decathlon
    Task01_BrainTumour dataset.

    Expected structure:

        Task01_BrainTumour/
        ├── dataset.json
        ├── imagesTr/
        │   ├── BRATS_001.nii.gz
        │   ├── BRATS_002.nii.gz
        │   └── ...
        ├── labelsTr/
        │   ├── BRATS_001.nii.gz
        │   ├── BRATS_002.nii.gz
        │   └── ...
        └── imagesTs/

    Each training image is expected to be a 4D NIfTI volume:

        (H, W, D, C)

    where C is the number of MRI modalities.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        expected_modalities: int = 4,
    ):
        self.root = Path(dataset_root)

        self.images_dir = self.root / "imagesTr"
        self.labels_dir = self.root / "labelsTr"
        self.dataset_json = self.root / "dataset.json"

        self.expected_modalities = expected_modalities

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> Dict:

        logger.info("=" * 70)
        logger.info("MSD DATASET VALIDATION")
        logger.info("=" * 70)

        report = {
            "dataset_root": str(self.root),
            "cases": 0,
            "images": 0,
            "labels": 0,
            "invalid_dimensions": 0,
            "wrong_modalities": 0,
            "missing_labels": 0,
            "corrupted_files": 0,
            "shape_mismatches": 0,
            "status": False,
        }

        self._validate_structure()

        cases = self._discover_cases()

        report["cases"] = len(cases)

        logger.info(
            f"Discovered {len(cases)} training cases."
        )

        for case_id in cases:

            logger.info(
                f"Validating {case_id}..."
            )

            image_path = (
                self.images_dir /
                f"{case_id}.nii.gz"
            )

            label_path = (
                self.labels_dir /
                f"{case_id}.nii.gz"
            )

            # ----------------------------------------------------------
            # Image
            # ----------------------------------------------------------

            if not image_path.exists():

                logger.warning(
                    f"{case_id}: image missing"
                )

                report["corrupted_files"] += 1
                continue

            report["images"] += 1

            try:

                image = nib.load(image_path)

            except Exception as exc:

                logger.error(
                    f"{case_id}: could not read image: {exc}"
                )

                report["corrupted_files"] += 1
                continue

            # ----------------------------------------------------------
            # Image dimensions
            # ----------------------------------------------------------

            if len(image.shape) != 4:

                logger.warning(
                    f"{case_id}: expected 4D image, "
                    f"got shape {image.shape}"
                )

                report["invalid_dimensions"] += 1

                continue

            modalities = image.shape[-1]

            if modalities != self.expected_modalities:

                logger.warning(
                    f"{case_id}: expected "
                    f"{self.expected_modalities} modalities, "
                    f"found {modalities}"
                )

                report["wrong_modalities"] += 1

            # ----------------------------------------------------------
            # Label
            # ----------------------------------------------------------

            if not label_path.exists():

                logger.warning(
                    f"{case_id}: label missing"
                )

                report["missing_labels"] += 1
                continue

            report["labels"] += 1

            try:

                label = nib.load(label_path)

            except Exception as exc:

                logger.error(
                    f"{case_id}: could not read label: {exc}"
                )

                report["corrupted_files"] += 1
                continue

            # ----------------------------------------------------------
            # Spatial shape consistency
            # ----------------------------------------------------------

            image_spatial_shape = image.shape[:3]
            label_shape = label.shape

            if image_spatial_shape != label_shape:

                logger.warning(
                    f"{case_id}: image/label shape mismatch "
                    f"{image_spatial_shape} != {label_shape}"
                )

                report["shape_mismatches"] += 1

        # --------------------------------------------------------------
        # Final status
        # --------------------------------------------------------------

        report["status"] = (
            report["cases"] > 0
            and report["images"] == report["cases"]
            and report["labels"] == report["cases"]
            and report["invalid_dimensions"] == 0
            and report["wrong_modalities"] == 0
            and report["missing_labels"] == 0
            and report["corrupted_files"] == 0
            and report["shape_mismatches"] == 0
        )

        self._print_report(report)

        return report

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def _validate_structure(self) -> None:

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist:\n{self.root}"
            )

        if not self.dataset_json.exists():
            raise FileNotFoundError(
                f"dataset.json not found:\n{self.dataset_json}"
            )

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"imagesTr directory not found:\n{self.images_dir}"
            )

        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f"labelsTr directory not found:\n{self.labels_dir}"
            )

        logger.info(
            "✓ Dataset structure found."
        )

    # ------------------------------------------------------------------
    # Case discovery
    # ------------------------------------------------------------------

    def _discover_cases(self) -> List[str]:

        labels = sorted(
            self.labels_dir.glob("*.nii.gz")
        )

        case_ids = []

        for path in labels:

            # Ignore macOS metadata files such as:
            #
            # ._BRATS_001.nii.gz

            if path.name.startswith("._"):
                continue

            case_id = path.name.replace(
                ".nii.gz",
                ""
            )

            case_ids.append(case_id)

        return case_ids

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _print_report(
        self,
        report: Dict,
    ) -> None:

        logger.info("")
        logger.info("=" * 70)
        logger.info("VALIDATION REPORT")
        logger.info("=" * 70)

        logger.info(
            f"Dataset root       : "
            f"{report['dataset_root']}"
        )

        logger.info(
            f"Cases              : "
            f"{report['cases']}"
        )

        logger.info(
            f"Image volumes      : "
            f"{report['images']}"
        )

        logger.info(
            f"Labels             : "
            f"{report['labels']}"
        )

        logger.info(
            f"Invalid dimensions : "
            f"{report['invalid_dimensions']}"
        )

        logger.info(
            f"Wrong modalities   : "
            f"{report['wrong_modalities']}"
        )

        logger.info(
            f"Missing labels     : "
            f"{report['missing_labels']}"
        )

        logger.info(
            f"Corrupted files    : "
            f"{report['corrupted_files']}"
        )

        logger.info(
            f"Shape mismatches   : "
            f"{report['shape_mismatches']}"
        )

        status = (
            "PASS"
            if report["status"]
            else "FAIL"
        )

        logger.info(
            f"Dataset status     : "
            f"{status}"
        )

        logger.info("=" * 70)
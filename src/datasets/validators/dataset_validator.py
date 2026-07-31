from pathlib import Path
from typing import Dict, List

import nibabel as nib

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetValidator:
    """
    Validator for Medical Segmentation Decathlon datasets.

    The validator checks:

    1. Required dataset files/directories exist.
    2. Training images exist.
    3. Training labels exist.
    4. Every case has all expected modalities.
    5. Every image can be opened as a NIfTI volume.
    6. Every label can be opened as a NIfTI volume.
    7. Image and label spatial shapes match.
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
        """
        Run all dataset validation checks.

        Returns
        -------
        Dict
            Validation report.
        """

        logger.info("=" * 70)
        logger.info("MSD DATASET VALIDATION")
        logger.info("=" * 70)

        report = {
            "dataset_root": str(self.root),
            "cases": 0,
            "images": 0,
            "labels": 0,
            "missing_modalities": 0,
            "missing_labels": 0,
            "corrupted_files": 0,
            "shape_mismatches": 0,
            "status": False,
        }

        self._validate_structure()

        cases = self._discover_cases()

        report["cases"] = len(cases)

        logger.info(f"Discovered {len(cases)} training cases.")

        for case_id in cases:

            logger.info(f"Validating {case_id}...")

            image_paths = self._get_case_images(case_id)
            label_path = self.labels_dir / f"{case_id}.nii.gz"

            report["images"] += len(image_paths)

            # ----------------------------------------------------------
            # Check modalities
            # ----------------------------------------------------------

            if len(image_paths) != self.expected_modalities:

                logger.warning(
                    f"{case_id}: expected "
                    f"{self.expected_modalities} modalities, "
                    f"found {len(image_paths)}"
                )

                report["missing_modalities"] += 1

            # ----------------------------------------------------------
            # Check label
            # ----------------------------------------------------------

            if not label_path.exists():

                logger.warning(
                    f"{case_id}: missing label"
                )

                report["missing_labels"] += 1

                continue

            report["labels"] += 1

            # ----------------------------------------------------------
            # Load images and label
            # ----------------------------------------------------------

            image_shapes = []

            for image_path in image_paths:

                try:

                    image = nib.load(image_path)

                    image_shapes.append(image.shape)

                except Exception as exc:

                    logger.error(
                        f"Could not read {image_path}: {exc}"
                    )

                    report["corrupted_files"] += 1

            try:

                label = nib.load(label_path)

            except Exception as exc:

                logger.error(
                    f"Could not read label {label_path}: {exc}"
                )

                report["corrupted_files"] += 1

                continue

            # ----------------------------------------------------------
            # Check image consistency
            # ----------------------------------------------------------

            if image_shapes:

                reference_shape = image_shapes[0]

                for shape in image_shapes[1:]:

                    if shape != reference_shape:

                        logger.warning(
                            f"{case_id}: modality shape mismatch "
                            f"{reference_shape} != {shape}"
                        )

                        report["shape_mismatches"] += 1

            # ----------------------------------------------------------
            # Check image-label spatial dimensions
            # ----------------------------------------------------------

            if image_shapes:

                reference_shape = image_shapes[0]

                if label.shape != reference_shape:

                    logger.warning(
                        f"{case_id}: image/label shape mismatch "
                        f"{reference_shape} != {label.shape}"
                    )

                    report["shape_mismatches"] += 1

        # --------------------------------------------------------------
        # Final status
        # --------------------------------------------------------------

        report["status"] = (
            report["cases"] > 0
            and report["missing_modalities"] == 0
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
        """
        Validate the basic MSD directory structure.
        """

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

        logger.info("✓ Dataset structure found.")

    # ------------------------------------------------------------------
    # Case discovery
    # ------------------------------------------------------------------

    def _discover_cases(self) -> List[str]:
        """
        Discover case IDs from labelsTr.

        Example:

            BRATS_001.nii.gz
            BRATS_002.nii.gz

        becomes:

            BRATS_001
            BRATS_002
        """

        labels = sorted(
            self.labels_dir.glob("*.nii.gz")
        )

        case_ids = [
            path.name.replace(".nii.gz", "")
            for path in labels
        ]

        return case_ids

    # ------------------------------------------------------------------
    # Image discovery
    # ------------------------------------------------------------------

    def _get_case_images(
        self,
        case_id: str,
    ) -> List[Path]:
        """
        Return all modalities belonging to one case.
        """

        images = sorted(
            self.images_dir.glob(
                f"{case_id}_*.nii.gz"
            )
        )

        return images

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
            f"Dataset root       : {report['dataset_root']}"
        )

        logger.info(
            f"Cases              : {report['cases']}"
        )

        logger.info(
            f"Image volumes      : {report['images']}"
        )

        logger.info(
            f"Labels             : {report['labels']}"
        )

        logger.info(
            f"Missing modalities : {report['missing_modalities']}"
        )

        logger.info(
            f"Missing labels     : {report['missing_labels']}"
        )

        logger.info(
            f"Corrupted files    : {report['corrupted_files']}"
        )

        logger.info(
            f"Shape mismatches   : {report['shape_mismatches']}"
        )

        status = (
            "PASS"
            if report["status"]
            else "FAIL"
        )

        logger.info(
            f"Dataset status     : {status}"
        )

        logger.info("=" * 70)
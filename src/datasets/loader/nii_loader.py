from pathlib import Path

import nibabel as nib
import numpy as np


class NiftiLoader:
    """
    Utility class for loading NIfTI volumes.
    """

    @staticmethod
    def load(path):

        path = Path(path)

        img = nib.load(path)

        return img.get_fdata()

    @staticmethod
    def affine(path):

        path = Path(path)

        return nib.load(path).affine

    @staticmethod
    def header(path):

        path = Path(path)

        return nib.load(path).header
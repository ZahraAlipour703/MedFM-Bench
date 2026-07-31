from pathlib import Path

from src.datasets.loaders.nii_loader import NiftiLoader

sample = Path(
    "datasets/raw/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz"
)

volume = NiftiLoader.load(sample)

print(volume.shape)
print(volume.dtype)
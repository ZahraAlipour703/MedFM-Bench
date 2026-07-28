from src.datasets import *
from src.registry import DATASETS

print(DATASETS.keys())

dataset = DATASETS.build(
    "brats",
    root_dir="datasets/raw"
)

print(dataset)
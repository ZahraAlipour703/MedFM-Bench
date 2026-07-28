from src.datasets import *
from src.registry.dataset_registry import build_dataset

dataset = build_dataset(
    "brats",
    root_dir="datasets/raw"
)

print(dataset)
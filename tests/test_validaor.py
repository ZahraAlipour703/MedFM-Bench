from src.datasets.validators.dataset_validator import DatasetValidator

validator = DatasetValidator(
    "datasets/raw/Task01_BrainTumour"
)

report = validator.validate()

print(report)
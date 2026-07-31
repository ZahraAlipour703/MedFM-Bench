from src.datasets.validators import DatasetValidator


def main():

    validator = DatasetValidator(
        "datasets/raw/Task01_BrainTumour"
    )

    report = validator.validate()

    if not report["status"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
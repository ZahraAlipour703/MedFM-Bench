from src.datasets.brats import BraTSDataset
from src.datasets.split import (
    split_dataset,
    save_split,
)


def main():

    dataset = BraTSDataset(
        "datasets/raw/Task01_BrainTumour"
    )

    split = split_dataset(
        dataset.samples,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )

    print("=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)

    for name, samples in split.items():

        print(
            f"{name:>5}: {len(samples)} patients"
        )

    train_ids = {
        x["patient_id"]
        for x in split["train"]
    }

    val_ids = {
        x["patient_id"]
        for x in split["val"]
    }

    test_ids = {
        x["patient_id"]
        for x in split["test"]
    }

    print("=" * 60)
    print("OVERLAP CHECK")
    print("=" * 60)

    print(
        "Train ∩ Val:",
        len(train_ids & val_ids),
    )

    print(
        "Train ∩ Test:",
        len(train_ids & test_ids),
    )

    print(
        "Val ∩ Test:",
        len(val_ids & test_ids),
    )

    save_split(
        split,
        "configs/splits/brats_seed42.json",
    )

    print("=" * 60)
    print("Saved:")
    print("configs/splits/brats_seed42.json")


if __name__ == "__main__":
    main()
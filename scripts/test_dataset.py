from src.datasets.brats import BraTSDataset


def main():

    dataset = BraTSDataset(
        "datasets/raw/Task01_BrainTumour"
    )

    print("=" * 60)
    print("DATASET TEST")
    print("=" * 60)

    print(f"Number of samples: {len(dataset)}")

    sample = dataset[0]

    print(f"Patient ID:       {sample['patient_id']}")
    print(f"Image shape:      {sample['image'].shape}")
    print(f"Image dtype:      {sample['image'].dtype}")
    print(f"Label shape:      {sample['label'].shape}")
    print(f"Label dtype:      {sample['label'].dtype}")

    print("=" * 60)


if __name__ == "__main__":
    main()
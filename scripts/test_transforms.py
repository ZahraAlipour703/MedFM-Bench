from src.datasets.brats import BraTSDataset
from src.transforms import (
    zscore_normalize,
    normalize_label,
    to_tensor,
)


def main():

    dataset = BraTSDataset(
        "datasets/raw/Task01_BrainTumour"
    )

    sample = dataset[0]

    image = sample["image"]
    label = sample["label"]

    print("=" * 60)
    print("RAW")
    print("=" * 60)

    print("Image:", image.shape, image.dtype)
    print("Label:", label.shape, label.dtype)

    image = zscore_normalize(image)
    label = normalize_label(label)

    image, label = to_tensor(
        image,
        label,
    )

    print("=" * 60)
    print("PROCESSED")
    print("=" * 60)

    print("Image:", image.shape, image.dtype)
    print("Label:", label.shape, label.dtype)

    print("=" * 60)
    print("IMAGE STATISTICS")
    print("=" * 60)

    print("Mean:", image.mean().item())
    print("Std :", image.std().item())

    print("=" * 60)
    print("LABEL CLASSES")
    print("=" * 60)

    print(
        "Unique labels:",
        label.unique().tolist()
    )


if __name__ == "__main__":
    main()
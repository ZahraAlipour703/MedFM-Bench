from torch.utils.data import DataLoader


def build_dataloader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
):
    """
    Build a PyTorch DataLoader.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
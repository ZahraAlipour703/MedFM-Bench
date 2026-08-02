from torch.utils.data import Dataset


class DatasetSubset(Dataset):
    """
    A lightweight subset wrapper for patient-level splits.
    """

    def __init__(
        self,
        dataset,
        samples,
    ):
        self.dataset = dataset
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        return self.dataset[
            self.dataset.samples.index(sample)
        ]
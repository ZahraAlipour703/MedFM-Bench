from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseMedicalDataset(Dataset, ABC):
    """
    Base class for all medical segmentation datasets.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass
from .base_dataset import BaseMedicalDataset
from src.registry.dataset_registry import register_dataset

@register_dataset("brats")
class BraTSDataset(BaseMedicalDataset):

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)
        self.samples = self.load_data()

    def load_data(self):
        return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pass
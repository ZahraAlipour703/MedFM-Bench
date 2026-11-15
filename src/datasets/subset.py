
from torch.utils.data import Dataset


class DatasetSubset(Dataset):
    """
    Efficient patient-level subset wrapper.

    Maps patient IDs to the original dataset indices once,
    rather than searching the dataset for every retrieval.
    """

    def __init__(self, dataset, samples):
        self.dataset = dataset
        self.samples = list(samples)

        patient_to_index = {}

        for index, sample in enumerate(dataset.samples):
            patient_id = sample["patient_id"]

            if patient_id in patient_to_index:
                raise ValueError(
                    f"Duplicate patient ID in dataset: {patient_id}"
                )

            patient_to_index[patient_id] = index

        selected_ids = [
            sample["patient_id"]
            for sample in self.samples
        ]

        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(
                "Duplicate patient IDs in subset."
            )

        missing = [
            patient_id
            for patient_id in selected_ids
            if patient_id not in patient_to_index
        ]

        if missing:
            raise ValueError(
                f"Patients not found in dataset: {missing[:10]}"
            )

        self.indices = [
            patient_to_index[patient_id]
            for patient_id in selected_ids
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

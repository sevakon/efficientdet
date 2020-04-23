import config as cfg

from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):
    def __init__(self):
        self.samples = list(range(cfg.NUM_EXAMPLES_PER_EPOCH))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_loader():
    dataset = DummyDataset()
    return DataLoader(dataset=dataset, batch_size=cfg.BATCH_SIZE)
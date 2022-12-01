from torch.utils.data import Dataset
import torch


class PandaSet(Dataset):
    """
    Dataset from pandas dataframe.
    Target column must be the last one:
        df[df.columns[:-1]] -> x
        df[df.columns[-1]] -> y
    """

    def __init__(self, data):
        super().__init__()

        x = data[data.columns[:-1]].values
        y = data[data.columns[-1]].values

        self.x = torch.tensor(x).to(torch.float32)
        self.y = torch.tensor(y).to(torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MyDataset(Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(x).to(torch.float32)
        self.y = torch.tensor(y).to(torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

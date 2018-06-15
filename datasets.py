import torch
import torchvision.datasets
from torch.utils.data import Dataset


class MnistFlattenedDataset(Dataset):
    def __init__(self, train):
        ds = torchvision.datasets.MNIST(
            './data/mnist',
            download=True,
            transform=None,
            train=train
        )

        data = ds.train_data.type(torch.FloatTensor) if train else ds.test_data.type(torch.FloatTensor)

        self.data = data.view(data.size()[0], -1)
        self.labels = ds.train_labels if train else ds.test_labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class MnistDataset(Dataset):
    def __init__(self, train):
        ds = torchvision.datasets.MNIST(
            './data/mnist',
            download=True,
            transform=None,
            train=train
        )

        self.data = ds.train_data if train else ds.test_data
        self.labels = ds.train_labels if train else ds.test_labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class FashionMnistDataset(Dataset):
    def __init__(self, train):
        ds = torchvision.datasets.FashionMNIST(
            './data/fashion-mnist',
            download=True,
            transform=None,
            train=train
        )

        self.data = ds.train_data if train else ds.test_data
        self.labels = ds.train_labels if train else ds.test_labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

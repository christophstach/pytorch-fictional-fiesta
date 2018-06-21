import math

import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MnistFlattenedDataset(Dataset):
    def __init__(self, train):
        ds = torchvision.datasets.MNIST(
            './data/mnist',
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            train=train
        )

        data = ds.train_data.type(torch.FloatTensor) if train else ds.test_data.type(torch.FloatTensor)

        self.data = data.view(data.size()[0], -1)
        self.labels = ds.train_labels if train else ds.test_labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


'''
'''


class MnistDataset(Dataset):
    def __init__(self, train):
        ds = torchvision.datasets.MNIST(
            './data/mnist',
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            train=train
        )

        self.data = ds.train_data.unsqueeze(1).type(torch.FloatTensor) if train \
            else ds.test_data.unsqueeze(1).type(torch.FloatTensor)
        self.labels = ds.train_labels if train else ds.test_labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


'''
'''


class FashionMnistDataset(Dataset):
    def __init__(self, train):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        ds = torchvision.datasets.FashionMNIST(
            './data/fashion-mnist',
            download=True,
            transform=transform,
            train=train
        )

        self.data = ds.train_data.unsqueeze(1).type(torch.FloatTensor) if train \
            else ds.test_data.unsqueeze(1).type(torch.FloatTensor)
        self.labels = ds.train_labels if train else ds.test_labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


'''
Dataset for prime numbers
'''


class PrimeNumberDataset(Dataset):
    def __init__(self, train):
        self.data = []
        self.labels = []

        if train:
            for i in range(100000):
                self.data.append([i])

                is_prime = self.is_prime(i)
                self.labels.append([
                    1 if not is_prime else 0,
                    1 if is_prime else 0
                ])

        else:
            for i in range(100000, 120000):
                self.data.append([i])

                is_prime = self.is_prime(i)
                self.labels.append([
                    1 if not is_prime else 0,
                    1 if is_prime else 0
                ])

        self.data = torch.FloatTensor(self.data)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def is_prime(self, number):
        if number == 0 or number == 1 or number == 2:
            return True
        else:
            prime = True
            until = int(math.ceil(math.sqrt(number))) + 1

            for i in range(2, until):
                if number % i == 0:
                    prime = False
                    break

            return prime

import torch.nn as nn
import torch.nn.functional as F

'''
A simple three fully connected layer network
'''


class ThreeLayerFullyConnectedNetwork(nn.Module):
    def __init__(self):
        super(ThreeLayerFullyConnectedNetwork, self).__init__()

        self.fc1 = nn.Linear(in_features=784, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=80)
        self.fc3 = nn.Linear(in_features=80, out_features=10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x


'''
The convolutional network from the pytorch tutorial
'''


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(84, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(-1, 16 * 4 * 4)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = F.softmax(x, dim=1)
        return x


'''
Neural network to calculate prime numbers 
'''


class PrimeNumberNetwork(nn.Module):
    def __init__(self):
        super(PrimeNumberNetwork, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=1, out_features=120),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=80),
            nn.Sigmoid()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=80, out_features=2),
            nn.Sigmoid(),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

import torch.nn as nn
import torch.nn.functional as F


class MnistFullyConnectedNet(nn.Module):
    def __init__(self):
        super(MnistFullyConnectedNet, self).__init__()

        self.fc1 = nn.Linear(in_features=784, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=80)
        self.fc3 = nn.Linear(in_features=80, out_features=10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x

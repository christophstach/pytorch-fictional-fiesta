import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from mnist_dataset import MnistDataset

print(torch.__version__)

train = datasets.MNIST('./data/mnist/train', download=True, transform=None, train=True)
test = datasets.MNIST('./data/mnist/test', download=True, transform=None, train=False)

train_x = train.train_data.type(torch.FloatTensor)
train_y = train.train_labels

test_x = test.test_data.type(torch.FloatTensor)
test_y = test.test_labels

train_x = train_x.view(train_x.size()[0], -1)
test_x = test_x.view(test_x.size()[0], -1)

train = MnistDataset(data=train_x, labels=train_y)
test = MnistDataset(data=test_x, labels=test_y)

train_loader = DataLoader(train, batch_size=1000)
test_loader = DataLoader(test, batch_size=1000)


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()

        self.fc1 = nn.Linear(in_features=784, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=80)
        self.fc3 = nn.Linear(in_features=80, out_features=10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x

    def start_training(self, optimizer, criterion, loader, epochs=100):
        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(loader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            print('Epoch:', epoch, 'Loss:', round(loss.item(), 5))

    def start_testing(self, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy: %d %%' % (100 * correct / total))


print('')
print('######################################')
print('')

net = MnistNet()
params = list(net.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

print(net)
print('Params[0]:', params[0].size())

print('')
print('######################################')
print('')
print('Starting training')

#net.start_training(criterion=criterion, optimizer=optimizer, loader=train_loader)
#net.start_testing(loader=test_loader)

print('Finished Training')

print('')
print('######################################')
print('')

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

# net_in = torch.randn((5, 784))
labels = test_y[0:1]
inputs = test_x[0:1]
outputs = net(inputs)
_, predicted = torch.max(outputs, 1)
# print('Input:', inputs)
print('Output:', torch.max(outputs))
print('Predicted: ', predicted)
print('labels: ', labels)

print('Cuda: ', torch.cuda.is_available())

import argparse
import os.path
import pickle
from statistics import mean

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import FashionMnistDataset
from models import ConvolutionalNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
args = parser.parse_args()

print('''

########################################################################################################################
# Loading data                                                                                                         #
########################################################################################################################

''')

train_dataset = FashionMnistDataset(train=True)
test_dataset = FashionMnistDataset(train=False)

train_loader = DataLoader(train_dataset, batch_size=160, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=True)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

print('''

########################################################################################################################
# Defining loss function and back propagation                                                                          #
########################################################################################################################

''')

net = ConvolutionalNetwork()
params = list(net.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(net)

print('''

########################################################################################################################
# Training                                                                                                             #
########################################################################################################################

''')

epochs = args.epochs
loss_history_file = './logs/mnist-convolutional-network.loss-history.pkl'
model_file = './models/mnist-convolutional-network.pt'

if os.path.isfile(loss_history_file) and os.path.getsize(loss_history_file) > 0:
    with open(loss_history_file, 'rb') as f:
        loss_history = pickle.load(f)
else:
    loss_history = []

if os.path.isfile(model_file) and os.path.getsize(model_file) > 0:
    state = torch.load(model_file)
    net.load_state_dict(state)

for epoch in tqdm(range(epochs), desc='Training'):
    losses_in_epoch = []
    for data in train_loader:
        inputs, labels = data
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses_in_epoch.append(loss.item())
    loss_history.append(mean(losses_in_epoch))

    if epoch % 5 == 0:
        torch.save(net.state_dict(), model_file)
        with open(loss_history_file, 'wb') as f:
            pickle.dump(loss_history, f)

torch.save(net.state_dict(), model_file)
with open(loss_history_file, 'wb') as f:
    pickle.dump(loss_history, f)

print('''

########################################################################################################################
# Testing                                                                                                              #
########################################################################################################################

''')

correct = 0
total = 0

class_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
class_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        for i, prediction in enumerate(predicted):
            total += 1
            class_total[labels[i]] += 1
            correct += 1 if prediction.item() == labels[i] else 0
            class_correct[labels[i]] += 1 if prediction.item() == labels[i] else 0

print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
for i, o in enumerate(class_correct):
    print(classes[i] + ':', '%.2f %%' % (100 * o / class_total[i]))

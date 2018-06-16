import argparse
import os.path
import pickle
from statistics import mean

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import PrimeNumberDataset
from models import PrimeNumberNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()

print('''

########################################################################################################################
# Loading data                                                                                                         #
########################################################################################################################

''')

train_dataset = PrimeNumberDataset(train=True)
test_dataset = PrimeNumberDataset(train=False)

train_loader = DataLoader(train_dataset, batch_size=2000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20000, shuffle=True)

print('''

########################################################################################################################
# Defining loss function and back propagation                                                                          #
########################################################################################################################

''')

net = PrimeNumberNetwork()
params = list(net.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
print(net)

print('''

########################################################################################################################
# Training                                                                                                             #
########################################################################################################################

''')

epochs = args.epochs
loss_history_file = './logs/prime-number-network.loss-history.pkl'
model_file = './models/prime-number-network.pt'

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
        loss = criterion(outputs, torch.max(labels, 1)[1])
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

for i in range(50):
    print(i, net(torch.FloatTensor([i])))

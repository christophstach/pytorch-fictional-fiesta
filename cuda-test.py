import math
import torch.cuda
from datasets import PrimeNumberDataset

print(torch.cuda.is_available())
print(torch.cuda.device_count())


ds_train = PrimeNumberDataset(train=True)
ds_test = PrimeNumberDataset(train=False)



print(len(ds_train.data))
print(len(ds_train.labels))


print(len(ds_test.data))
print(len(ds_test.labels))



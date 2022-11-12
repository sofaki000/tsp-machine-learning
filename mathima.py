import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def my_fun(x): # edw einai to diktyo
  return (x+3)**2

x_0 = torch.tensor(1.0, requires_grad=True)

y = my_fun(x_0)

print(y)

y.backward() #kanoume backpropagation: klish x_o ws pros y (partial paragogos)dy/dx_0

lr=0.001
x_1 = x_0 - lr *x_0.grad

x_new = torch.tensor(1.0, requires_grad=True)

for i in range(10):
  #get the data
  x = torch.tensor(x_new.data, requires_grad=True)
  # feed forward the network and calculate the loss
  y = my_fun(my_fun(x))
  # do back propagation
  y.backward() # thelei na kanei optimize-> na mikrinei to y!briskei oles tis merikes paragogous ws pros y
  #optimize the loss
  x_new = x-lr * x.grad #edw einai stohastic gradient descent

  print(y)

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1,16,3)
        self.conv1 = nn.Conv2d(16,32,3)

        self.fc1 = nn.Linear(5*5*32, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self,x):
        # channels*width*height : 1*28*28-> 16*26*(26)
        # first convolutional
        x = self.conv1(x)
        x = F.relu(x)
        x= F.max_pool2d(x,2)

        # sec convolutional
        x = self.conv2(x)#32*11*11
        x = F.relu(x)
        x = F.max_pool2d(x,2)


        # flattening layer
        x = x.view(-1, 5*5*32)

        #twra tha perasoum apo fully connected later
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


net = Lenet()
input = torch.rand(1,1,28,28)
y = net(input)


print(y.size()) #1*10-> 1 input, exodos apo 10 neurwnes

# dummy loss, elaxistopoiei exodo diktyou
loss = torch.sum(y)
loss.backward()

# loading data

import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)

trainLoader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

from tqdm import tqdm #dinei grafiko periballon
for epoch in range(2):
    epoch_loss=0
    for data,label in tqdm(trainLoader):
        output = net(data)
        current_loss = loss(output, label)
        current_loss.backwards() #back propagate loss
        # update parameters
        optimizer.zero_grad()
        optimizer.step()
        epoch_loss+= current_loss.item()
    print(f'epoch {epoch}, loss={current_loss}')
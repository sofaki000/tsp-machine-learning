# this program outputs a sequence of possible solutions for an input tour for the tsp.
# no optimization is being made. the results are purely a feasible solution (visiting each city only once)
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

learning_rate = 1e-3
train_size = 10
num_nodes = 5
tsp_data = np.random.rand(train_size, num_nodes, 2)  # syntetagmenes twn nodes mas
n_epochs = 4


class Model(nn.Module): #gyrnaei probability distribution gia to epomeno city pou tha episkeftoume
  def __init__(self, n_feature, n_hidden, output_size):
      super(Model, self).__init__()
      self.l1 = nn.Linear(n_feature, n_hidden)
      self.l2 = nn.Linear(n_hidden, output_size)
  def forward(self, current_city):
      encoded = self.l1(current_city)
      probs = self.l2(encoded)
      return probs


model = Model(n_feature=2, n_hidden=128, output_size=num_nodes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
current_city = tsp_data[:, 0, :]  # gia kathe sample, pare tis syntetagmenes tou prwtou city

# mask is used to not revisit same city
mask = torch.zeros(train_size, num_nodes, dtype=torch.bool)

tours = []
for sample_idx in range(train_size):
    tours.append([])

for idx in range(num_nodes):  # gia kathe sample input tha bgaloume ena tour
    current_city = torch.tensor(current_city, dtype=torch.float32)
    output = model(current_city)

    probabilities = output.clone()
    probabilities[mask] = 100000.0

    masked_probabilities = torch.tensor(probabilities.detach(), requires_grad=True)
    masked_probabilities = F.log_softmax(masked_probabilities.contiguous(), -1)  # (bs, L)

    sampler = torch.distributions.Categorical(masked_probabilities)
    chosen_city = sampler.sample()
    mask[[i for i in range(train_size)], chosen_city] = True

    # gia kathe sample, tha prosthesoume sto tour tou to chosen city
    for sample_idx in range(train_size):
        tours[sample_idx].append(chosen_city[sample_idx].item())
    print(f"Choosen city: {chosen_city}")


for i in range(train_size):
    # gia kathemia diadromh
    for node_idx in range(num_nodes):
        print(tours[i][node_idx]),
    print("--------------------")
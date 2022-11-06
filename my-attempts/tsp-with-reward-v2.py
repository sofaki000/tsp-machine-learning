# this program outputs a very little optimized tsp solution.
# As a loss we use the mean of distances travelled from current city to next (mean of all distances
# for each sample)
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from plotUtilities import save_plot_with_y_axis
from reward import get_distance_of_two_nodes_given_coordinates

learning_rate = 0.05
train_size = 100
num_nodes = 3
tsp_data = np.random.rand(train_size, num_nodes, 2)  # syntetagmenes twn nodes mas
n_epochs = 200


class Model(nn.Module): #gyrnaei probability distribution gia to epomeno city pou tha episkeftoume
  def __init__(self, n_feature, n_hidden, output_size):
      super(Model, self).__init__()
      self.l1 = nn.LSTM(n_feature, n_hidden)
      self.l2 = nn.Linear(n_hidden, output_size)
  def forward(self, current_city):
      output, (hidden_state, cell_state) = self.l1(current_city)
      probs = self.l2(output)
      return probs


model = Model(n_feature=2, n_hidden=128, output_size=num_nodes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay =0.01)
current_city = tsp_data[:, 0, :]  # gia kathe sample, pare tis syntetagmenes tou prwtou city

# mask is used to not revisit same city
mask = torch.zeros(train_size, num_nodes, dtype=torch.bool)

tours = []
for sample_idx in range(train_size):
    tours.append([])

mean_losses_per_epoch = []
for epoch in range(n_epochs):

    losses_per_epoch = []
    for idx in range(num_nodes):  # gia kathe sample input tha bgaloume ena tour
        current_city = torch.Tensor(current_city)
        output = model(current_city)

        probabilities = output.clone()
        probabilities[mask] = 100000.0

        masked_probabilities = probabilities.clone().detach().requires_grad_(True)
        masked_probabilities = F.log_softmax(masked_probabilities.contiguous(), -1)  # (bs, L)

        sampler = torch.distributions.Categorical(masked_probabilities)
        chosen_city = sampler.sample()

        # we mark each chosen visit as visited so we dont visit it in the future
        mask[[i for i in range(train_size)], chosen_city] = True

        # we calculate the distance from current city to proposed city for each sample tour
        distances_traveled = []
        for tour_idx in range(train_size):
            coordinates_of_current_city = current_city[tour_idx].detach().numpy()
            coordinates_of_city_to_visit = tsp_data[tour_idx, chosen_city[tour_idx],:]
            coordinates = [coordinates_of_current_city, coordinates_of_city_to_visit]
            distance_between_current_and_prev_city = get_distance_of_two_nodes_given_coordinates(coordinates)
            distances_traveled.append(distance_between_current_and_prev_city)
            # gia kathe sample, tha prosthesoume sto tour tou to chosen city
            tours[tour_idx].append(chosen_city[tour_idx].item())

        loss = torch.tensor(distances_traveled, dtype=torch.float32, requires_grad=True)
        # print(f"loss: {loss.detach().item():.3f}")
        losses_per_epoch.append(loss.detach().mean().item())
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
    mean_loss = np.mean(losses_per_epoch)
    print(f"Mean loss at epoch {epoch}: {mean_loss:.2f}")
    mean_losses_per_epoch.append(mean_loss)


title = f"Num epochs: {n_epochs}, lr:{learning_rate}, nodes:{num_nodes}"
file_name ="results/mean_losses_per_epoch_"
save_plot_with_y_axis(mean_losses_per_epoch,title, file_name,xLabel="Epochs", yLabel="Mean loss")


cities = []
for c in range(num_nodes):
    cities.append(c)

# for tour_idx in range(train_size):
#     # gia kathemia diadromh
#     for node_idx in range(num_nodes):
#         print(tours[tour_idx][node_idx], end =" ")
#     coordinates_for_current_tour= tsp_data[tour_idx]
#     distance_traveled_at_tour = get_distance_from_coordinate_pairs(coordinates=coordinates_for_current_tour, cities=cities, tour=tours[tour_idx])
#     print(f"tour length {distance_traveled_at_tour}")
#     print("--------------------")


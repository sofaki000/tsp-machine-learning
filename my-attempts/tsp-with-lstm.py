from tqdm import tqdm
import numpy as np
import torch
from model import EncoderDecoderModel
from torch import optim
from torch.utils.data import DataLoader
from neural_combinatorial_optimization_rl.tsp_heuristic import get_ref_reward
from plotUtilities import save_plot_with_y_axis, save_plot_with_multiple_functions_in_same_figure
from tsp import TSPDataset
from tsp_heuristics import tsp_dynamic_programing_solution

learning_rate = 0.001
train_size = 500
test_size = 200
num_nodes = 5
train_dataset = TSPDataset(num_nodes, train_size)
test_dataset = TSPDataset(num_nodes, test_size)
n_epochs = 200
batch_size = 1

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True )

hidden_size = 128
embedding_size = 128
beta= 0.9
model = EncoderDecoderModel(hidden_size=hidden_size,  embedding_size=embedding_size)
experiments_mean_losses = []
titles = []

optimizer = optim.Adam(model.parameters(), lr=3.0 * 1e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay =0.01)

# mask is used to not revisit same city
mask = torch.zeros(train_size, num_nodes, dtype=torch.bool)

tours = []
for sample_idx in range(train_size):
    tours.append([])

model.train()

mean_losses_per_epoch = []
moving_avg = torch.zeros(train_size)

# generating first baseline
for (indices, sample_batch) in tqdm(train_data_loader):
    rewards, _, _ = model(sample_batch)
    moving_avg[indices] = rewards


average_dist_per_epoch = []
for epoch in range(n_epochs):
    hidden = torch.zeros(train_size, num_nodes)
    losses_per_epoch = []

    distance_travelled_per_epoch = []
    for batch_idx, (indices, cities_in_batch) in enumerate(train_data_loader):
        rewards, log_probs, action = model(cities_in_batch)

        moving_avg[indices] = moving_avg[indices] * beta + rewards * (1.0 - beta)

        advantage = rewards - moving_avg[indices]

        log_probs = torch.sum(log_probs, dim=-1)
        log_probs[log_probs < -100] = -100
        loss = (advantage * log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        optimizer.step()

        losses_per_epoch.append(loss.detach().mean())
        average_distance_traveled_per_batch = rewards.mean().detach().numpy()
        distance_travelled_per_epoch.append(average_distance_traveled_per_batch)

    print(f'Average distance per epoch: {np.mean(distance_travelled_per_epoch):.3f}')
    average_dist_per_epoch.append(np.mean(distance_travelled_per_epoch))
model_path = "models/encoderDecoder"
torch.save(model.state_dict(), model_path)

title = f"Encoder Decoder model: Epochs={n_epochs}, lr={learning_rate}, n={num_nodes}"
file_name ="results/EncDecoderAttention/neural_combinatorial_my_impl"
save_plot_with_y_axis(y_axis=distance_travelled_per_epoch, title=title, file_name=file_name,xLabel="Epochs", yLabel="Average distance travelled")


# testing
# Comparing with a heuristic in test phase
trained_model = EncoderDecoderModel(hidden_size=hidden_size,  embedding_size=embedding_size)
trained_model.load_state_dict(torch.load(model_path))


# Calculating heuristics
heuristic1_distance = torch.zeros(test_size)
heuristic2_distance = np.zeros(test_size)

# for i in range(test_size):
#     heuristic2_distance.append([])

for i, pointset in tqdm(test_dataset):
    heuristic2_distance[i] = tsp_dynamic_programing_solution( pointset )
    heuristic1_distance[i] = get_ref_reward(pointset)


rl_distances_travelled = []
for i, batch in test_data_loader:
    R, _, _ = model(batch)
    rl_distances_travelled.append(R.mean().detach().item())
    # print( f"[at epoch {epoch}]RL model generates {(R / heuristic_distance).mean().detach().numpy():0.2f} time worse solution than heuristics")
    # print("AVG R", R.mean().detach().numpy())


save_plot_with_multiple_functions_in_same_figure([heuristic1_distance, heuristic2_distance, rl_distances_travelled], ["Heuristic 1", "Heuristic 2","RL solution"], "results/comparison_with_heuristics/rl_comparison_with_heuristic", title)

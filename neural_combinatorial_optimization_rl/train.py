import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from solver import  solver_RNN
from tsp import TSPDataset
from tsp_heuristic import get_ref_reward

use_cuda = False
seq_len= 5 #30
num_epochs = 10
num_tr_dataset = 50
num_test_dataset = 20
embedding_size = 128 #den paizei gia 256!
hidden_size = 128
batch_size =1 # 64
grad_clip = 1.5
beta=0.9

if __name__ =="__main__":
    if use_cuda:
        use_pin_memory = True
    else:
        use_pin_memory = False
    train_dataset = TSPDataset(seq_len, num_tr_dataset)
    test_dataset = TSPDataset(seq_len, num_test_dataset)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_pin_memory)

    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,  pin_memory=use_pin_memory)
    eval_loader = DataLoader(test_dataset, batch_size=num_test_dataset, shuffle=False)

    # Calculating heuristics
    heuristic_distance = torch.zeros(num_test_dataset)
    for i, pointset in tqdm(test_dataset):
        heuristic_distance[i] = get_ref_reward(pointset)


    model = solver_RNN(embedding_size, hidden_size, seq_len, 2, 10)

    if use_cuda:
        model = model.cuda()
    print(f'Num of params:{sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])}')
    optimizer = optim.Adam(model.parameters(), lr=3.0 * 1e-4)

    # Train loop
    moving_avg = torch.zeros(num_tr_dataset)
    if use_cuda:
        moving_avg = moving_avg.cuda()

    #generating first baseline
    for (indices, sample_batch) in tqdm(train_data_loader):
        if use_cuda:
            sample_batch = sample_batch.cuda()
        rewards, _, _ = model(sample_batch)
        moving_avg[indices] = rewards

    #Training
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (indices, sample_batch) in enumerate(train_data_loader):
            if use_cuda:
                sample_batch.cuda()
            rewards, log_probs, action = model(sample_batch) #to sample batch exei ola ta cities
            # tou kathe sample h ena ena ta cities tou sample?-> ola ta cities tou sample
            moving_avg[indices] = moving_avg[indices] * beta + rewards * (1.0 - beta)
            advantage = rewards - moving_avg[indices]
            log_probs = torch.sum(log_probs, dim=-1)
            log_probs[log_probs < -100] = -100
            loss = (advantage * log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()


        model.eval()
        ret = []
        for i, batch in eval_loader:

            if use_cuda:
                batch = batch.cuda()
            R, _, _ = model(batch)
        print(f"[at epoch {epoch}]RL model generates { (R / heuristic_distance).mean().detach().numpy():0.2f} time worse solution than heuristics" )
        print("AVG R", R.mean().detach().numpy())
        model.train()
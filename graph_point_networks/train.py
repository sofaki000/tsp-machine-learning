import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from graph_point_networks.GraphPointerNetworkModel import GPN
from plotUtilities import plot_model_with_me

if __name__ == "__main__":

    # args
    size = 10
    n_epoch = 100
    batch_size = 512
    train_size = 2000
    validation_size = 1000
    learn_rate = 1e-3
    save_root = './model/gpn_tsp' + str(size) + '.pt'


    model = GPN(n_feature=2, n_hidden=128)#.cuda()

    # load model
    # model = torch.load(save_root).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    lr_decay_step = 2500
    lr_decay_rate = 0.96
    opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step * 1000, lr_decay_step), gamma=lr_decay_rate)

    # validation data
    validation_data = np.random.rand(validation_size, size, 2)

    C = 0  # baseline
    R = 0  # reward

    # R_mean = []
    # R_std = []
    for epoch in range(n_epoch):
        for i in tqdm(range(train_size)):
            optimizer.zero_grad()

            X = np.random.rand(batch_size, size, 2)

            X = torch.Tensor(X)#.cuda()

            mask = torch.zeros(batch_size, size)#.cuda()

            R = 0
            logprobs = 0
            reward = 0

            Y = X.view(batch_size, size, 2)
            x = Y[:, 0, :]
            h = None
            c = None

            for k in range(size):
                output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
                sampler = torch.distributions.Categorical(output)
                idx = sampler.sample()  # now the idx has B elements

                Y1 = Y[[i for i in range(batch_size)], idx.data].clone()
                if k == 0:
                    Y_ini = Y1.clone()
                if k > 0:
                    reward = torch.norm(Y1 - Y0, dim=1)

                Y0 = Y1.clone()
                x = Y[[i for i in range(batch_size)], idx.data].clone()
                R += reward

                TINY = 1e-15
                logprobs += torch.log(output[[i for i in range(batch_size)], idx.data] + TINY)
                mask[[i for i in range(batch_size)], idx.data] += -np.inf

            R += torch.norm(Y1 - Y_ini, dim=1)

            # self-critic base line
            mask = torch.zeros(batch_size, size)#.cuda()

            C = 0
            baseline = 0

            Y = X.view(batch_size, size, 2)
            x = Y[:, 0, :]
            h = None
            c = None

            for k in range(size):

                output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)

                # sampler = torch.distributions.Categorical(output)
                # idx = sampler.sample()         # now the idx has B elements
                idx = torch.argmax(output, dim=1)  # greedy baseline

                Y1 = Y[[i for i in range(batch_size)], idx.data].clone()
                if k == 0:
                    Y_ini = Y1.clone()
                if k > 0:
                    baseline = torch.norm(Y1 - Y0, dim=1)

                Y0 = Y1.clone()
                x = Y[[i for i in range(batch_size)], idx.data].clone()

                C += baseline
                mask[[i for i in range(batch_size)], idx.data] += -np.inf

            C += torch.norm(Y1 - Y_ini, dim=1)

            gap = (R - C).mean()
            loss = ((R - C - gap) * logprobs).mean()

            loss.backward()

            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type=2)
            optimizer.step()
            opt_scheduler.step()

            if i % 50 == 0:
                print(f"epoch:{epoch}, batch:{i}/{train_size}, reward:{R.mean().item()}")
                # R_mean.append(R.mean().item())
                # R_std.append(R.std().item())

                # greedy validation
                tour_len = 0
                X = validation_data
                X = torch.Tensor(X) #.cuda()

                mask = torch.zeros(validation_size, size)#.cuda()

                R = 0
                logprobs = 0
                Idx = []
                reward = 0

                Y = X.view(validation_size, size, 2)  # to the same batch size
                x = Y[:, 0, :]
                h = None
                c = None

                for k in range(size):

                    output, h, c, hidden_u = model(x=x, X_all=X, h=h, c=c, mask=mask)

                    sampler = torch.distributions.Categorical(output)
                    # idx = sampler.sample()
                    idx = torch.argmax(output, dim=1)
                    Idx.append(idx.data)

                    Y1 = Y[[i for i in range(validation_size)], idx.data]

                    if k == 0:
                        Y_ini = Y1.clone()
                    if k > 0:
                        reward = torch.norm(Y1 - Y0, dim=1)

                    Y0 = Y1.clone()
                    x = Y[[i for i in range(validation_size)], idx.data]

                    R += reward

                    mask[[i for i in range(validation_size)], idx.data] += -np.inf

                R += torch.norm(Y1 - Y_ini, dim=1)
                tour_len += R.mean().item()
                print('validation tour length:', tour_len)

        print('save model to: ', save_root)
        torch.save(model, save_root)
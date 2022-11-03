import torch
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from reward import get_distance_from_coordinate_pairs

def train_model_that_returns_indexes(model, X, Y, batch_size, n_epochs):
    model.train()
    optimizer = optim.Adam(model.parameters())
    samples_size = X.size(0)
    sequence_length = X.size(1)
    losses = []
    for epoch in range(n_epochs + 1):
        mean_loss_per_epoch =0
        # for i in range(len(train_batches))
        for i in range(0, samples_size - batch_size, batch_size): #louparoume ta batch sizes
            x = X[i:i + batch_size]  # (bs, sequence_length)
            y = Y[i:i + batch_size]  # (bs, sequence_length)

            (probs, tour) = model(x)  # (bs, M, L)
            outputs = probs.view(-1, sequence_length)  # (bs*M, L)

            y = y.reshape(-1)  # (bs*M)

            loss = F.nll_loss(outputs, y)
            mean_loss_per_epoch = loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print('epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))
            test(model, X, Y)
        losses.append(mean_loss_per_epoch)
    indexes = []
    for i in range(n_epochs+1):
        indexes.append(i)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.plot(indexes, losses)
    plt.show()



def train_model_that_returns_probabilities_sequence(model, X, Y, batch_size, n_epochs):
    model.train()
    optimizer = optim.Adam(model.parameters())

    samples_size = X.size(0)
    sequence_length = X.size(1)
    losses = []
    for epoch in range(n_epochs + 1):
        mean_loss_per_epoch =0
        # for i in range(len(train_batches))
        for i in range(0, samples_size - batch_size, batch_size):
            x = X[i:i + batch_size]  # (bs, sequence_length)
            y = Y[i:i + batch_size]  # (bs, sequence_length)

            probs = model(x)  # (bs, M, L)
            outputs = probs.view(-1, sequence_length)  # (bs*M, L)

            y = y.reshape(-1)  # (bs*M)

            loss = F.nll_loss(outputs, y)
            mean_loss_per_epoch = loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print('epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))
            test(model, X, Y)
        losses.append(mean_loss_per_epoch)
    indexes = []
    for i in range(n_epochs+1):
        indexes.append(i)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.plot(indexes, losses)
    plt.show()


def test(model, X, Y, coordinates = None):
    probs = model(X)  # (bs, M, L)
    if len(probs.size())==2:
        _v, result_sequence = torch.max(probs, 1)
    else:
        _v, result_sequence = torch.max(probs, 2)

    if coordinates is not None:
        cities = []
        for i in range(X.shape[1]):
            cities.append(i)
        for i in range(len(result_sequence)):
            # gia kathe test sample koitame pws phgame sygkritika me ton heuristic algorithmo
            distance_of_heuristic = get_distance_from_coordinate_pairs(coordinates[0], cities ,Y[i])
            distance_of_model = get_distance_from_coordinate_pairs(coordinates[0], cities, result_sequence[i])
            print(f'Tour suggested by heuristic: {Y[i]}')
            print(f"Distance traveled (heuristic solution): {distance_of_heuristic:.2f}")
            print(f'Tour suggested by model: {result_sequence[i]}')
            print(f"Distance traveled (model solution): {distance_of_model:.2f}")

    correct_count = sum([1 if torch.equal(ind.data, y.data) else 0 for ind, y in zip(result_sequence, Y)])
    if(len(X)==0):
        return
    print(f'Acc: {(correct_count / len(X) * 100):.2f}% ({correct_count}/{ len(X)})')

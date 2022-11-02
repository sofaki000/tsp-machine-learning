import torch
from torch import optim
import torch.nn.functional as F


# from pointer_network import PointerNetwork
def train(model, X, Y, batch_size, n_epochs):
    model.train()
    optimizer = optim.Adam(model.parameters())
    samples_size = X.size(0)
    sequence_length = X.size(1)

    for epoch in range(n_epochs + 1):
        # for i in range(len(train_batches))
        for i in range(0, samples_size - batch_size, batch_size):
            x = X[i:i + batch_size]  # (bs, sequence_length)
            y = Y[i:i + batch_size]  # (bs, sequence_length)

            probs = model(x)  # (bs, M, L)
            outputs = probs.view(-1, sequence_length)  # (bs*M, L)

            y = y.reshape(-1)  # (bs*M)
            loss = F.nll_loss(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print('epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))
            test(model, X, Y)


def test(model, X, Y):
    probs = model(X)  # (bs, M, L)
    _v, indices = torch.max(probs, 2)
    correct_count = sum([1 if torch.equal(ind.data, y.data) else 0 for ind, y in zip(indices, Y)])
    print('Acc: {:.2f}% ({}/{})'.format(correct_count / len(X) * 100, correct_count, len(X)))

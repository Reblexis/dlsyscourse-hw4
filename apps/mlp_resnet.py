import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    block = nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU()
    )
    return block


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes)
    )


def epoch(dataloader, model, opt=None):
    np.random.seed(4)

    model.eval() if opt is None else model.train()

    losses = []
    total = 0
    correct = 0

    for X, y in dataloader:
        X = ndl.ops.reshape(X, (X.shape[0], -1))
        logits = model(X)
        loss = nn.SoftmaxLoss()(logits, y)
        losses.append(loss.numpy())
        total += y.shape[0]
        correct += np.sum(np.argmax(logits.numpy(), axis=1) == y.numpy())

        if opt is not None:
            loss.backward()
            opt.step()

    return (total-correct) / total, float(np.mean(losses))


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    mnist_dataset_train = ndl.data.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", f"{data_dir}/train-labels-idx1-ubyte.gz")
    mnist_dataset_test = ndl.data.MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(mnist_dataset_train, batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(mnist_dataset_test, batch_size, shuffle=False)

    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    last_stats = None
    for epoch_num in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)
        test_err, test_loss = epoch(test_dataloader, model)
        last_stats = (train_err, train_loss, test_err, test_loss)
        print(last_stats)

    return last_stats


if __name__ == "__main__":
    train_mnist(data_dir="../data")
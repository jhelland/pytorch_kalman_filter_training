import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from models import DFN
from optimizers import EKF


if __name__ == "__main__":
    stdev = 0.05
    U = np.arange(-10, 10, 0.2)
    Y = np.exp(-U**2) + 0.5 * np.exp(-(U-3)**2) + np.random.normal(0, stdev, len(U))
    U, Y = torch.from_numpy(U.astype("float32")), torch.from_numpy(Y.astype("float32"))

    epochs_sgd = 100
    model_sgd = DFN([1, 10, 1], activation=nn.Sigmoid)

    epochs_ekf = 100
    model_ekf = DFN([1, 10, 1], activation=nn.Sigmoid)

    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.05)
    optimizer_ekf = EKF(model_ekf.parameters(), lr=1, num_outputs=1, P=0.5, Q=0, R=0.1+stdev**2)

    losses_sgd = []
    model_sgd.train()

    losses_ekf = []
    model_ekf.train()
    for epoch in range(max(epochs_sgd, epochs_ekf)):
        rand_idx = np.random.permutation(len(U))
        U_shuffled = U[rand_idx].reshape(U.numel(), 1)
        Y_shuffled = Y[rand_idx].reshape(Y.numel(), 1)

        for i, (u, y) in enumerate(zip(U_shuffled, Y_shuffled)):
            if epoch < epochs_sgd:
                # train with SGD
                optimizer_sgd.zero_grad()
                h = model_sgd(u)
                loss_sgd = F.mse_loss(h, y)
                loss_sgd.backward()
                optimizer_sgd.step()
                losses_sgd.append(loss_sgd.data)

            if epoch < epochs_ekf:
                # train with global EKF
                optimizer_ekf.zero_grad()
                h = model_ekf(u)
                losses_ekf.append(F.mse_loss(h, y).data)

                # udpate EKF variables
                optimizer_ekf.y = y
                optimizer_ekf.h = h
                optimizer_ekf.step()  # automatically does back propagation

        print("\ntraining: epoch {} | SGD loss {:.4f} | EKF loss {:.4f}".format(epoch, losses_sgd[-1], losses_ekf[-1]))

    # evaluation
    model_sgd.eval()
    model_ekf.eval()
    with torch.no_grad():
        X = np.arange(-15, 15, 0.01, dtype="float32")
        plt.suptitle("Data fit", fontsize=22)
        plt.plot(X, np.exp(-X**2) + 0.5*np.exp(-(X-3)**2), c='b', alpha=0.5, lw=2, ls="-", label="True")
        plt.plot(X, model_sgd(torch.from_numpy(X).reshape(len(X), 1)).numpy(), c='k', lw=3, ls="--", label="Adam: {} epochs".format(epochs_sgd))
        plt.plot(X, model_ekf(torch.from_numpy(X).reshape(len(X), 1)).numpy(), c='g', lw=3, label="EKF: {} epochs".format(epochs_ekf))
        plt.scatter(U, Y, c='b', s=5)
        plt.grid(True)
        plt.legend(fontsize=22)
        plt.show()

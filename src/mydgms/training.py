# 誤差と訓練アルゴリズム
from dataclasses import dataclass
import numpy as np
from .neuralnet import MyNeuralNet, Tensor, Loss


def train_mgd(init_net: MyNeuralNet, loss: Loss, X: Tensor, y: Tensor) -> MyNeuralNet:
    """
    ニューラルネットワークを訓練する。
    最適化アルゴリズムはミニバッチ勾配降下法を採用する。
    """
    n_epochs = 1000
    n_samples = X.shape[0]
    batch_size = 20
    learning_rate = 0.01

    # メインループ
    net = init_net
    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(n_samples)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, n_samples, batch_size):
            xi = X_shuffled[i : i + batch_size]
            yi = y_shuffled[i : i + batch_size]
            grads = loss.gradient(net=net, X=xi, y=yi)
            net = net.update(grads, learning_rate=learning_rate)  # 更新

        loss_all = loss.eval(net=net, X=X, y=y)
        print(f"{epoch=}, {loss_all=}")
    return net

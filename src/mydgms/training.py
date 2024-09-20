# 誤差と訓練アルゴリズム
from dataclasses import dataclass
import numpy as np
from .neuralnet import MyNeuralNet, Tensor, Loss
from .generative import MyBinaryEnergyBasedModel, LogLoss


def create_mini_batches(X: Tensor, y: Tensor, batch_size: int):
    """ミニバッチを生成する関数"""
    indices = np.random.permutation(X.shape[0])
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    mini_batches = [
        (X_shuffled[i : i + batch_size], y_shuffled[i : i + batch_size]) for i in range(0, X.shape[0], batch_size)
    ]
    return mini_batches


def train_mgd(
    init_net: MyNeuralNet,
    loss: Loss,
    X: Tensor,
    y: Tensor,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    tolerance: float,
    X_test: Tensor = None,
    y_test: Tensor = None,
) -> MyNeuralNet:
    """
    ニューラルネットワークを訓練する。
    最適化アルゴリズムはミニバッチ勾配降下法を採用する。
    """
    n_samples = X.shape[0]

    # メインループ
    net = init_net
    previous_loss = float("inf")
    for epoch in range(n_epochs):
        mini_batches = create_mini_batches(X=X, y=y, batch_size=batch_size)

        # 各ミニバッチごとのループ
        for mini_batch in mini_batches:
            X_mini, y_mini = mini_batch

            # 勾配計算
            grads = loss.gradient(net=net, X=X_mini, y=y_mini)

            # 勾配にしたがってパラメータの更新ステップを決定
            params_step = [{param: -learning_rate * grad_param for param, grad_param in grad.items()} for grad in grads]

            net = net.update(params_step=params_step)  # 更新

        # 損失の変化量を計算
        train_loss = loss.eval(net=net, X=X, y=y)
        loss_change = abs(previous_loss - train_loss)
        previous_loss = train_loss

        # 1エポックごとに損失を表示
        if epoch % 5 == 0:
            test_loss = None
            if isinstance(X_test, Tensor):
                test_loss = loss.eval(net=net, X=X_test, y=y_test)
                print(f"{epoch=}, {train_loss=:.4f}, {test_loss=: .4f}, {loss_change=:.6f}")
            else:
                print(f"{epoch=}, {train_loss=:.4f}, {loss_change=:.6f}")

        # 収束判定
        if loss_change < tolerance:
            print(f"Converged at Epoch {epoch + 1}, Loss: {train_loss:.4f}")
            break
    return net


def create_mini_batches_X(X: Tensor, batch_size: int):
    """ミニバッチを生成する関数"""
    indices = np.random.permutation(X.shape[0])
    X_shuffled = X[indices]
    mini_batches = [X_shuffled[i : i + batch_size] for i in range(0, X.shape[0], batch_size)]
    return mini_batches


def train_generative_mgd(
    init_ebm: MyBinaryEnergyBasedModel,
    X: Tensor,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    tolerance: float,
    X_test: Tensor = None,
) -> MyBinaryEnergyBasedModel:
    """
    ニューラルネットワークを訓練する。
    最適化アルゴリズムはミニバッチ勾配降下法を採用する。
    """
    n_samples = X.shape[0]

    loss = LogLoss()

    # メインループ
    ebm = init_ebm
    previous_loss = float("inf")
    for epoch in range(n_epochs):
        mini_batches = create_mini_batches_X(X=X, batch_size=batch_size)

        # 各ミニバッチごとのループ
        for mini_batch in mini_batches:
            X_mini = mini_batch

            # 勾配計算
            grads = loss.gradient(ebm=ebm, X=X_mini)

            # 勾配にしたがってパラメータの更新ステップを決定
            params_step = [{param: -learning_rate * grad_param for param, grad_param in grad.items()} for grad in grads]

            ebm = ebm.update(params_step=params_step)  # 更新

        # 損失の変化量を計算
        train_loss = loss.eval(ebm=ebm, X=X)
        loss_change = abs(previous_loss - train_loss)
        previous_loss = train_loss

        # 1エポックごとに損失を表示
        if epoch % 10 == 0:
            test_loss = None
            if isinstance(X_test, Tensor):
                test_loss = loss.eval(ebm=ebm, X=X_test)
                print(f"{epoch=}, {train_loss=:.4f}, {test_loss=: .4f}, {loss_change=:.6f}")
            else:
                print(f"{epoch=}, {train_loss=:.4f}, {loss_change=:.6f}")

        # 収束判定
        if loss_change < tolerance:
            print(f"Converged at Epoch {epoch + 1}, Loss: {train_loss:.4f}")
            break
    return ebm

# ニューラルネットワークモデル
# 参考: https://qiita.com/miya_ppp/items/fd916da9da5578185bc8

from dataclasses import dataclass, field
import numpy as np

Tensor = np.ndarray


@dataclass
class Layer:
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def backward(self, x: Tensor, din: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError()


@dataclass
class ReLU(Layer):
    """
    ReLU関数
    """

    def forward(self, x: Tensor) -> Tensor:
        return np.maximum(0, x)

    def backward(self, x: Tensor, din: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """
        ReLU層の逆伝播
        Parameters
        ----------
        x : Tensor
            N x d
            この層の入力であって、ニューラルネットワーク全体の入力ではない点に注意
        din: Tensor
            N x d

        Returns
        -------
        dout: Tensor
            N x d
        """
        return din * np.where(x > 0, 1, 0), {}


@dataclass
class Dense(Layer):
    """
    全結合関数

    W: Tensor
        d x M 行列
    b: Tensor
        M ベクトル
    """

    W: Tensor
    b: Tensor

    def __post_init__(self):
        if self.W.ndim != 2:
            raise ValueError(f"{self.W.ndim=} は2である必要があります。")
        if self.W.shape[1] != self.b.shape[0]:
            raise ValueError(f"{self.W.shape=}, {self.b.shape=}")

    def forward(self, x: Tensor) -> Tensor:
        return np.dot(x, self.W) + self.b

    def backward(self, x: Tensor, din: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Parameters
        ----------
        x : Tensor
            N x d
            この層の入力であって、ニューラルネットワーク全体の入力ではない点に注意
        din: Tensor
            N x M

        Returns
        -------
        dout: Tensor
            N x d
        grad_W
            d x M
        grad_b
            M
        """
        grad_W = np.dot(x.T, din)  # d x M
        grad_b = din.sum(axis=0)  # M

        dout = np.dot(din, self.W.T)  # N x d
        return dout, {"W": grad_W, "b": grad_b}


@dataclass
class MyNeuralNet:
    layers: list[Layer]
    d_input: int
    d_output: int
    n_depths: int = field(init=False)

    def __post_init__(self):
        self.n_depths = len(self.layers)
        self.d_input
        self.d_output

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """
        ニューラルネットワークの順伝播
        Parameters
        ----------
        x: Tensor
            N x d_input

        Return
        ------
        Tensor
            N x M
        list[Tensor]
            中間ユニットの値
        """
        if x.shape[1] != self.d_input:
            raise ValueError(f"{x.shape[1]=}は {self.d_input=}と等しい必要があります。")

        x_l = x
        us = []
        for layer in self.layers:
            us.append(x_l)
            x_l = layer.forward(x_l)
        return x_l, us

    def gradient(self, x: Tensor, din: Tensor) -> tuple[Tensor, list[dict[str, Tensor]]]:
        """
        x: Tensor
        din: Tensor
        """
        if x.shape[1] != self.d_input:
            raise ValueError(f"{x.shape[1]=}は {self.d_input=}と等しい必要があります。")
        if din.shape[1] != self.d_output:
            raise ValueError(f"{din.shape[1]=}は {self.d_output=}と等しい必要があります。")
        if x.shape[0] != din.shape[0]:
            raise ValueError(f"{x.shape[0]=}は {din.shape[0]=}と等しい必要があります。")

        y, us = self.forward(x=x)  # まず順伝播で途中の計算履歴を取得
        grads = []
        dout, _ = self.layers[-1].backward(x=us[-1], din=din)  # 最終層の勾配dy/duを計算

        # 誤差逆伝播法で各パラメータの勾配を計算
        for layer, u in zip(self.layers[::-1], us[::-1]):
            din, grad = layer.backward(x=u, din=din)
            grads.append(grad)
        return dout, grads[::-1]


@dataclass
class Loss:
    pass


@dataclass
class SquaredLoss(Loss):
    def eval(self, net: MyNeuralNet, X: Tensor, y: Tensor) -> float:
        # 損失関数の計算
        y_pred, _ = net.forward(x=X)
        return np.sum((y_pred - y) ** 2) / 2

    def backward(self, net: MyNeuralNet, X: Tensor, y: Tensor) -> Tensor:
        # 損失関数の逆伝播
        y_pred, _ = net.forward(x=X)
        return y_pred - y

    def gradient(self, net: MyNeuralNet, X: Tensor, y: Tensor) -> list[dict[str, Tensor]]:
        din = self.backward(net=net, X=X, y=y)
        _, grads = net.gradient(x=X, din=din)
        return grads

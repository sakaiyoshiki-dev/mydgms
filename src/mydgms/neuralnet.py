# ニューラルネットワークモデル
# 参考: https://qiita.com/miya_ppp/items/fd916da9da5578185bc8

from dataclasses import dataclass, field
import numpy as np
from typing import Self

Tensor = np.ndarray


@dataclass
class Layer:
    """ニューラルネットワークの層の基底クラス"""

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def backward(self, x: Tensor, dout: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError()

    def backward_numerical(self, x: Tensor, dout: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError()

    def update(self, params_step: dict[str, Tensor]) -> Self:
        raise NotImplementedError()


@dataclass
class ReLU(Layer):
    """
    ReLU関数
    """

    def forward(self, x: Tensor) -> Tensor:
        """ReLU関数の計算
        Parameters
        ----------
        x: Tensor
            N x d

        Return
        Tensor
            N x d
        """
        return np.maximum(0, x)

    def backward(self, x: Tensor, dout: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """
        ReLU層の逆伝播
        Parameters
        ----------
        x : Tensor
            N x d
            この層の入力であって、ニューラルネットワーク全体の入力ではない点に注意
        dout: Tensor
            N x d
            上の層から降ってきた勾配dy/du

        Returns
        -------
        Tensor
            N x d
            下の層へ降ろす勾配dy/dx
        dict[str, Tensor]
            この層のパラメータについての勾配
        """
        return dout * np.where(x > 0, 1, 0), {}

    def backward_numerical(self, x: Tensor, dout: Tensor, eps: float) -> tuple[Tensor, dict[str, Tensor]]:
        """数値部分で勾配を計算

        主にテスト用"""
        # 数値微分による勾配を確認
        grad_numerical = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                original_value = x[i, j]

                x[i, j] = original_value + eps
                output_plus = self.forward(x)

                x[i, j] = original_value - eps
                output_minus = self.forward(x)

                grad_numerical[i, j] = np.sum(output_plus - output_minus) / (2 * eps)

                # 元に戻す
                x[i, j] = original_value

        return dout * grad_numerical, {}

    def update(self, params_step: dict[str, Tensor]) -> Self:
        """パラメータをparams_stepの方向に更新する"""
        return self


@dataclass
class Softmax(Layer):
    def softmax(self, x: Tensor) -> Tensor:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 数値安定化のために最大値を引く
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.softmax(x=x)

    def backward(self, x: Tensor, dout: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        s = self.forward(x=x)
        dint = s * (dout - np.sum(dout * s, axis=1, keepdims=True))
        # 参考: https://chatgpt.com/share/66e8e572-b6ec-800e-9092-5e86d21683f4 で計算
        return dint, {}

    def backward_numerical(self, x: Tensor, dout: Tensor, eps: float) -> tuple[Tensor, dict[str, Tensor]]:
        """数値部分で勾配を計算

        主にテスト用

        Parameters
        ----------
        x : Tensor
            N x d
            この層の入力であって、ニューラルネットワーク全体の入力ではない点に注意
        dout: Tensor
            N x d
            上の層から降ってきた勾配dy/du

        Returns
        -------
        Tensor
            N x d
            下の層へ降ろす勾配dy/dx
        dict[str, Tensor]
            この層のパラメータについての勾配
        """
        N, d = x.shape
        jacobian = np.zeros((N, d, d))

        for i in range(d):  # 各入力要素に対して数値微分を計算
            x_plus = np.copy(x)
            x_minus = np.copy(x)
            x_plus[:, i] += eps
            x_minus[:, i] -= eps
            # 各行にソフトマックスを適用した結果の差分を取る
            jacobian[:, :, i] = (self.forward(x_plus) - self.forward(x_minus)) / (2 * eps)

        din = np.zeros_like(dout)
        for n in range(N):
            dout_n = dout[n, :]  # 1 x d
            jac_n = jacobian[n, :, :]  # d x d

            din[n, :] = np.dot(dout_n, jac_n)
        return din, {}

    def update(self, params_step: dict[str, Tensor]) -> Self:
        return self


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
        """全結合層の計算
        Parameters
        ----------
        x: Tensor
            N x d

        Return
        ------
        Tensor
            N x M
        """
        return np.dot(x, self.W) + self.b

    def backward(self, x: Tensor, dout: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Parameters
        ----------
        x : Tensor
            N x d
            この層の入力であって、ニューラルネットワーク全体の入力ではない点に注意
        dout: Tensor
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
        grad_W = np.dot(x.T, dout)  # d x M
        grad_b = dout.sum(axis=0)  # M

        din = np.dot(dout, self.W.T)  # N x d
        return din, {"W": grad_W, "b": grad_b}

    def update(self, params_step: dict[str, Tensor]) -> Self:
        """パラメータをparams_stepの方向に更新した新しい層を返す"""
        return Dense(
            W=self.W + params_step["W"],
            b=self.b + params_step["b"],
        )

    @classmethod
    def init(cls: Self, d, M, rand=True) -> Self:
        if rand:
            W = 0.01 * np.random.randn(d, M)  # 初期値は小さいほうがよい（ゼロつくより）
            b = 0.01 * np.random.randn(M)
        else:
            W = np.zeros((d, M))  # 初期値が0はまずい（ゼロつくより）
            b = np.zeros(M)
        return cls(W=W, b=b)


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

    def gradient(self, x: Tensor, dout: Tensor) -> tuple[Tensor, list[dict[str, Tensor]]]:
        """
        x: Tensor
        dout: Tensor
        """
        if x.shape[1] != self.d_input:
            raise ValueError(f"{x.shape[1]=}は {self.d_input=}と等しい必要があります。")
        if dout.shape[1] != self.d_output:
            raise ValueError(f"{dout.shape[1]=}は {self.d_output=}と等しい必要があります。")
        if x.shape[0] != dout.shape[0]:
            raise ValueError(f"{x.shape[0]=}は {dout.shape[0]=}と等しい必要があります。")

        y, us = self.forward(x=x)  # まず順伝播で途中の計算履歴を取得
        grads = []
        din, _ = self.layers[-1].backward(x=us[-1], dout=dout)  # 最終層の勾配dy/duを計算

        # 誤差逆伝播法で各パラメータの勾配を計算
        for layer, u in zip(self.layers[::-1], us[::-1]):
            dout, grad = layer.backward(x=u, dout=dout)
            grads.append(grad)
        return din, grads[::-1]

    def update(self, params_step: list[dict[str, Tensor]]) -> Self:
        """
        パラメータ群を更新する
        いまのインスタンスは更新せず、新しいインスタンスを生成する。
        """
        new_layers = [layer.update(param_step) for layer, param_step in zip(self.layers, params_step)]
        return MyNeuralNet(layers=new_layers, d_input=self.d_input, d_output=self.d_output)


@dataclass
class Loss:
    def eval():
        pass

    def gradient():
        pass


@dataclass
class SquaredLoss(Loss):
    def eval(self, net: MyNeuralNet, X: Tensor, y: Tensor) -> float:
        """二乗誤差関数の計算
        Parameters
        ----------
        net: MyNeuralNet
        X: Tensor
        y: Tensor

        Return
        ------
        float
            二乗誤差の数値
        """
        y_pred, _ = net.forward(x=X)
        return np.sum((y_pred - y) ** 2) / y.shape[0] / 2

    def backward(self, net: MyNeuralNet, X: Tensor, y: Tensor) -> Tensor:
        """二乗誤差関数の逆伝播
        Parameters
        ----------
        net: MyNeuralNet
        X: Tensor
        y: Tensor

        Return
        ------
        Tensor
            ニューラルネットワークに渡すdy/du
        """
        y_pred, _ = net.forward(x=X)
        return (y_pred - y) / y.shape[0]

    def gradient(self, net: MyNeuralNet, X: Tensor, y: Tensor) -> list[dict[str, Tensor]]:
        """二乗誤差関数の勾配計算
        Parameters
        ----------
        net: MyNeuralNet
        X: Tensor
        y: Tensor

        Return
        ------
        list[dict[str, Tensor]
            各層の、パラメータごとの勾配ベクトル
        """
        din = self.backward(net=net, X=X, y=y)
        _, grads = net.gradient(x=X, dout=din)
        return grads

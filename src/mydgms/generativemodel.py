from dataclasses import dataclass, field
import numpy as np
from itertools import product
from typing import Self

from .neuralnet import MyNeuralNet, Tensor, Loss


@dataclass
class BaseGenerativeModel:
    def prob(self, x: Tensor) -> np.ndarray:
        raise NotImplementedError()


@dataclass
class MyBinaryEnergyBasedModel(BaseGenerativeModel):
    """0/1バイナリの入力変数に対するエネルギーベースモデル"""

    energy_func: MyNeuralNet
    d_input: int
    partition: float = field(init=False)

    def __post_init__(self):
        if self.d_input != self.energy_func.d_input:
            raise ValueError(f"{self.d_input=} != {self.energy_func.d_input=}")
        self.partition = self.calc_patrition()

    def calc_patrition(self) -> float:
        """分配関数を計算"""
        x_all = np.array([pattern for pattern in product([0, 1], repeat=2)])
        all_energy, _ = self.energy_func.forward(x=x_all)

        return np.exp(-all_energy).sum()

    def prob(self, x: Tensor) -> Tensor:
        """
        確率計算

        Parameter
        ---------
        x: Tensor

        """
        energy_val, _ = self.energy_func.forward(x)
        return np.exp(-np.squeeze(energy_val)) / self.partition

    def logprob(self, x: Tensor) -> Tensor:
        """
        対数尤度計算

        Parameter
        ---------
        x: Tensor

        """
        energy_val, _ = self.energy_func.forward(x)
        return -energy_val - np.log(self.partition)

    def update(self, params_step: list[dict[str, Tensor]]) -> Self:
        """
        パラメータ群を更新する
        いまのインスタンスは更新せず、新しいインスタンスを生成する。
        """
        new_net = self.energy_func.update(params_step=params_step)
        return MyBinaryEnergyBasedModel(energy_func=new_net, d_input=self.d_input)


class LogLoss(Loss):
    """負の対数損失"""

    def eval(self, ebm: MyBinaryEnergyBasedModel, X: Tensor) -> float:
        energy_val, _ = ebm.energy_func.forward(x=X)
        return (energy_val + np.log(ebm.partition)).sum() / X.shape[0]

    def backward(self, ebm: MyBinaryEnergyBasedModel, X: Tensor) -> Tensor:
        raise NotImplementedError()

    def gradient(self, ebm: MyBinaryEnergyBasedModel, X: Tensor) -> list[dict[str, Tensor]]:
        energy_grads: list[dict[str, Tensor]] = ebm.energy_func.gradient(x=X)
        grad_1st = []
        for grad in energy_grads:
            grad_1st.append({param: grad_p.sum(axis=0) / X.shape[0] for param, grad_p in grad.items()})
        return grad_1st

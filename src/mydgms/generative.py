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

    def grads_log_partition(self) -> list[dict[str, Tensor]]:
        """分配関数の勾配を直接計算"""
        x_all = np.array([pattern for pattern in product([0, 1], repeat=2)])
        prob_all = self.prob(x=x_all)

        grads_ret = [{} for _ in range(len(self.energy_func.layers))]  # 先に箱を用意しておく
        for x, prob in zip(x_all, prob_all):  # 全パターンに対して
            grads_energy: list[dict[str, Tensor]] = self.energy_func.gradient(x=np.array([x]))
            for i, grad_energy in enumerate(grads_energy):
                for param_name, grad in grad_energy.items():
                    if param_name not in grads_ret[i]:
                        grads_ret[i][param_name] = 0
                    grads_ret[i][param_name] += prob * grad
        return grads_ret


@dataclass
class MyEnergyBasedModel(BaseGenerativeModel):
    """一般のエネルギーベースモデル

    分配関数の計算が難しいため、サンプリングで近似していく必要がある"""

    pass


class LogLoss(Loss):
    """負の対数損失"""

    def eval(self, ebm: MyBinaryEnergyBasedModel, X: Tensor) -> float:
        energy_val, _ = ebm.energy_func.forward(x=X)
        return (energy_val + np.log(ebm.partition)).sum() / X.shape[0]

    def gradient(self, ebm: MyBinaryEnergyBasedModel, X: Tensor) -> list[dict[str, Tensor]]:
        grads_energy: list[dict[str, Tensor]] = ebm.energy_func.gradient(x=X)
        grads_log_partition: list[dict[str, Tensor]] = ebm.grads_log_partition()

        grads_ret = []
        for grad_energy, grad_part in zip(grads_energy, grads_log_partition):
            grad_ret = {}
            for param_name in grad_energy.keys():
                grad_ret[param_name] = grad_energy[param_name] / X.shape[0] - grad_part[param_name]
            grads_ret.append(grad_ret)
        return grads_ret

from dataclasses import dataclass, field
import numpy as np
from itertools import product
from typing import Self

from .neuralnet import MyNeuralNet, Tensor, Loss


@dataclass
class MyBinaryEnergyBasedModel:
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
class MyEnergyBasedModel:
    """一般のエネルギーベースモデル

    分配関数の計算が難しいため、サンプリングで近似していく必要がある"""

    energy_func: MyNeuralNet
    d_input: int

    def update(self, params_step: list[dict[str, Tensor]]) -> Self:
        new_net = self.energy_func.update(params_step=params_step)
        return MyEnergyBasedModel(energy_func=new_net, d_input=self.d_input)


class LangevinMCSampler:
    """ランジュバンモンテカルロ法によるサンプリング器"""

    def sample_from_ebm(self, ebm: MyEnergyBasedModel) -> Tensor:
        """EBMを前提にサンプリングする

        EBMでなくスコア関数が陽に与えられたときは別"""

        # ebm.energy_func (MyNeuralNet) に入力についての微分をさせる必要がある。
        dout = np.ones((1, ebm.energy_func.d_output)) / 1  # dout = [1/N,...,1/N]
        score_func = lambda x: ebm.energy_func.backward(x=x, dout=dout)[0]  # TODO: スコア関数が適切に実装されていない

        x_0 = np.random.normal(loc=0, scale=1, size=ebm.d_input)
        step_size = 50
        x_t = x_0
        eta = 0.1
        sigma = 1
        for t in range(step_size):
            x_t = x_t - eta * score_func(x=np.array([x_t])) + np.random.normal(loc=0, scale=sigma, size=ebm.d_input)

        raise x_t


class LogLoss:
    """負の対数損失"""

    def eval(self, ebm: MyBinaryEnergyBasedModel, X: Tensor) -> float:
        energy_val, _ = ebm.energy_func.forward(x=X)
        return (energy_val + np.log(ebm.partition)).sum() / X.shape[0]

    def gradient(self, ebm: MyBinaryEnergyBasedModel, X: Tensor) -> list[dict[str, Tensor]]:
        grads_energy: list[dict[str, Tensor]] = ebm.energy_func.gradient(x=X)
        grads_log_partition: list[dict[str, Tensor]] = ebm.grads_log_partition()

        # 勾配の統合
        grads_ret = []
        for grad_energy, grad_part in zip(grads_energy, grads_log_partition):
            grad_ret = {}
            for param_name in grad_energy.keys():
                grad_ret[param_name] = grad_energy[param_name] - grad_part[param_name]
            grads_ret.append(grad_ret)
        return grads_ret


@dataclass
class CDLoss:
    """いわゆるContrastive Divergence損失

    参考: [Tutorial 8: Deep Energy-Based Generative Models](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html)
    """

    sampler: LangevinMCSampler
    # サンプリング器は損失関数に付随するものとする

    def eval(self, ebm: MyEnergyBasedModel, X: Tensor) -> float:
        """
        CD損失 = エネルギー関数のデータ平均 - エネルギー関数のモデルサンプル平均
        """
        energies_data, _ = ebm.energy_func.forward(x=X)  # サンプル数×1

        X_model = self.sampler.sample_from_ebm(ebm=ebm)
        energies_model, _ = ebm.energy_func.forward(x=X_model)  # サンプル数×1

        return energies_data.mean() - energies_model.mean()

    def gradient(self, ebm: MyEnergyBasedModel, X: Tensor) -> list[dict[str, Tensor]]:
        """
        CD損失の勾配 = エネルギー関数勾配のデータ平均 - エネルギー関数勾配のモデルサンプル平均
        """
        # データパート
        grads_energy_data: list[dict[str, Tensor]] = ebm.energy_func.gradient(x=X)

        # モデルパート
        X_model = self.sampler.sample_from_ebm(ebm=ebm)
        grads_energy_model: list[dict[str, Tensor]] = ebm.energy_func.gradient(x=X_model)

        # 勾配の統合
        grads_ret = []
        for grad_data, grad_model in zip(grads_energy_data, grads_energy_model):
            grad_ret = {param_name: grad_data[param_name] - grad_model[param_name] for param_name in grad_data.keys()}
            grads_ret.append(grad_ret)
        return grads_ret

from dataclasses import dataclass, field
import numpy as np
from itertools import product

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
        return np.exp(-energy_val) / self.partition

    def logprob(self, x: Tensor) -> Tensor:
        """
        対数尤度計算

        Parameter
        ---------
        x: Tensor

        """
        energy_val, _ = self.energy_func.forward(x)
        return -energy_val - np.log(self.partition)


class LogLoss(Loss):
    """負の対数損失"""

    def eval(self, ebm: MyBinaryEnergyBasedModel, X: Tensor) -> float:
        energy_val, _ = ebm.energy_func.forward(x=X)
        return energy_val + np.log(ebm.partition)

    def backward(self, ebm: MyBinaryEnergyBasedModel, X: Tensor) -> Tensor:
        raise NotImplementedError()

    def gradient(self, ebm: MyBinaryEnergyBasedModel, X: Tensor) -> list[dict[str, Tensor]]:
        raise NotImplementedError()

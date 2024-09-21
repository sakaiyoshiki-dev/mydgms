# 深層生成モデルのテスト
import numpy as np
import pytest
from mydgms.neuralnet import MyNeuralNet, ReLU, Dense, Loss, Tensor
from mydgms.training import train_generative_mgd
from mydgms.generative import MyBinaryEnergyBasedModel


def test_BinaryEnergyBasedModel_確率確認():
    # 準備
    x = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    energy_func = MyNeuralNet(
        layers=[
            Dense(W=np.array([[1, 2], [3, 4]]), b=np.array([0.5, 0.7])),
            ReLU(),
            Dense(W=np.array([[0.5], [1.0]]), b=np.array([-3.0])),
        ],
        d_input=2,
        d_output=1,
    )
    ebm = MyBinaryEnergyBasedModel(energy_func=energy_func, d_input=2)

    # 実行
    probs = ebm.prob(x=x)

    # 検証
    assert probs.sum() == pytest.approx(1.0)
    expected = np.array([9.203804e-01, 3.761384e-03, 7.554943e-02, 3.087532e-04])
    np.testing.assert_allclose(probs, expected, atol=1e-5)


def test_BinaryEnergyBasedModel_学習():
    x = np.array([[0, 0]] * 500 + [[1, 0]] * 100 + [[1, 1]] * 400)
    energy_func = MyNeuralNet(
        layers=[
            Dense.init(d=2, M=4, seed=1234),
            ReLU(),
            Dense.init(d=4, M=1, seed=1234),
        ],
        d_input=2,
        d_output=1,
    )
    init_ebm = MyBinaryEnergyBasedModel(energy_func=energy_func, d_input=2)

    trained_ebm: MyBinaryEnergyBasedModel = train_generative_mgd(
        init_ebm=init_ebm,
        X=x,
        n_epochs=10000,
        batch_size=100,
        learning_rate=0.01,
        tolerance=1e-7,
    )

    assert trained_ebm.prob(x=np.array([[0, 0]])) == pytest.approx(0.5, abs=1e-2)
    assert trained_ebm.prob(x=np.array([[1, 0]])) == pytest.approx(0.1, abs=1e-2)
    assert trained_ebm.prob(x=np.array([[1, 1]])) == pytest.approx(0.4, abs=1e-2)

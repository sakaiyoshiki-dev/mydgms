# 深層生成モデルのテスト
import numpy as np
import pytest
from mydgms.neuralnet import MyNeuralNet, ReLU, Dense, Loss, Tensor
from mydgms.training import train_generative_mgd
from mydgms.generativemodel import MyBinaryEnergyBasedModel


def test_BinaryEnegeyBasedModel_確率確認():
    # 準備
    x = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    enegey_func = MyNeuralNet(
        layers=[
            Dense(W=np.array([[1, 2], [3, 4]]), b=np.array([0.5, 0.7])),
            ReLU(),
            Dense(W=np.array([[0.5], [1.0]]), b=np.array([-3.0])),
        ],
        d_input=2,
        d_output=1,
    )
    ebm = MyBinaryEnergyBasedModel(energy_func=enegey_func, d_input=2)

    # 実行
    probs = ebm.prob(x=x)

    # 検証
    assert probs.sum() == pytest.approx(1.0)
    expected = np.array([[9.203804e-01, 3.761384e-03, 7.554943e-02, 3.087532e-04]]).T
    np.testing.assert_allclose(probs, expected, atol=1e-5)


# def test_BinaryEnegeyBasedModel_学習():
#     x = np.array(
#         [
#             [0, 0],
#             [0, 0],
#             [0, 0],
#             [0, 0],
#             [0, 0],
#             [1, 0],
#             [1, 1],
#             [1, 1],
#             [1, 1],
#             [1, 1],
#         ]
#     )
#     enegey_func = MyNeuralNet(
#         layers=[
#             Dense.init(d=2, M=2, rand=True),
#             ReLU(),
#             Dense.init(d=2, M=1, rand=True),
#         ],
#         d_input=2,
#         d_output=1,
#     )
#     init_ebm = MyBinaryEnergyBasedModel(energy_func=enegey_func, d_input=2)

#     trained_ebm: MyBinaryEnergyBasedModel = train_generative_mgd(
#         init_ebm=init_ebm,
#         X=x,
#         n_epochs=1000,
#         batch_size=20,
#         learning_rate=0.01,
#         tolerance=1e-7,
#     )

#     assert trained_ebm.prob(x=np.array([[0, 0]])) == pytest.approx(0.5)
#     assert trained_ebm.prob(x=np.array([[1, 1]])) == pytest.approx(0.4)

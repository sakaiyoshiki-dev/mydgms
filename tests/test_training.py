# 訓練アルゴリズムのテスト

import numpy as np

from mydgms.neuralnet import MyNeuralNet, Dense, ReLU, SquaredLoss
from mydgms.training import train_mgd


def test_ミニバッチ勾配法():
    # 準備
    X = np.array([[0, 0.5, 1.0, 1.5, 2.0], [2.0, 0.5, 1.0, 1.5, 3.0]])
    y = np.array([[-3], [4]])
    init_net = MyNeuralNet(
        layers=[
            Dense(W=np.array([[-2, -1], [-1, -5], [0, 3], [1, -1], [2, -4]]), b=np.array([-1, 2])),
            ReLU(),
            Dense(W=np.array([[-2], [-1]]), b=np.array([0])),
        ],
        d_input=5,
        d_output=1,
    )
    loss = SquaredLoss()

    # 実行
    trained_net: MyNeuralNet = train_mgd(init_net=init_net, loss=loss, X=X, y=y)

    # 検証
    assert loss.eval(net=trained_net, X=X, y=y) <= 0.001

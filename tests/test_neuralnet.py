# 素朴なニューラルネットワークをテスト
import numpy as np
import pytest

from mydgms.neuralnet import MyNeuralNet, ReLU, Dense, Softmax, SquaredLoss


def test_ReLU層の数値検証():
    # N: 2, d: 5, M: 5
    relu = ReLU()
    x = np.array([[0, 0.5, 1.0, 1.5, -2.0], [2.0, 0.5, -1.0, 1.5, 0.0]])

    y = relu.forward(x=x)  # NxM = 2x5
    np.testing.assert_allclose(y, np.array([[0, 0.5, 1.0, 1.5, 0.0], [2.0, 0.5, 0.0, 1.5, 0.0]]))


def test_ReLU層の勾配検証():
    # N: 2, d: 5, M: 5
    relu = ReLU()
    x = np.array([[0, 0.5, 1.0, 1.5, -2.0], [2.0, 0.5, -1.0, 1.5, 0.0]])

    din = np.array([[0, 0.5, 1.0, 1.5, -2.0], [2.0, 0.5, -1.0, 1.5, 0.0]])  # NxM = 2x5

    y, _ = relu.backward(x=x, din=din)  # NxM = 2x5

    np.testing.assert_allclose(y, np.array([[0, 0.5, 1.0, 1.5, 0.0], [2.0, 0.5, 0.0, 1.5, 0.0]]))


def test_Dense1層の数値検証():
    # N: 2, d: 5, M: 2
    dense = Dense(W=np.array([[-2, -1], [-1, -5], [0, 3], [1, -1], [2, -4]]), b=np.array([-1, 2]))
    x = np.array([[0, 0.5, 1.0, 1.5, 2.0], [2.0, 0.5, 1.0, 1.5, 0.0]])

    y = dense.forward(x=x)  # NxM = 2x2
    np.testing.assert_allclose(y, np.array([[4.0, -7.0], [-4.0, -1.0]]))


def test_Dense1層の勾配検証():
    # N: 2, d: 5, M: 2
    dense = Dense(W=np.array([[-2, -1], [-1, -5], [0, 3], [1, -1], [2, -4]]), b=np.array([-1, 2]))
    x = np.array([[0, 0.5, 1.0, 1.5, 2.0], [2.0, 0.5, 1.0, 1.5, 0.0]])

    din = np.array([[-3.0, 3.0], [10.0, -10.0]])  # NxM
    dout, grads = dense.backward(x=x, din=din)
    # dout: Nxd
    # grads["W"]: dxM
    # grads["b"]: M

    np.testing.assert_allclose(dout, np.array([[3.0, -12.0, 9.0, -6.0, -18.0], [-10.0, 40.0, -30.0, 20.0, 60.0]]))
    np.testing.assert_allclose(
        grads["W"], np.array([[20.0, -20.0], [3.5, -3.5], [7.0, -7.0], [10.5, -10.5], [-6.0, 6.0]])
    )
    np.testing.assert_allclose(grads["b"], np.array([7.0, -7.0]))


def test_Softmax1層の数値検証():
    # N: 2, d: 5, M: 5
    softmax = Softmax()
    x = np.array([[0, 0.5, 1.0, 1.5, -2.0], [2.0, 0.5, -1.0, 1.5, 0.0]])

    y = softmax.forward(x=x)  # NxM = 2x5
    np.testing.assert_allclose(
        y,
        np.array(
            [[0.10016, 0.165136, 0.272263, 0.448886, 0.013555], [0.496331, 0.110746, 0.024711, 0.30104, 0.067171]]
        ),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        y.sum(axis=1),
        np.array([1, 1]),
        atol=1e-5,
    )


def test_Softmax1層の勾配検証():
    # N: 2, d: 5, M: 5
    softmax = Softmax()
    x = np.array([[0, 0.5, 1.0, 1.5, -2.0], [2.0, 0.5, -1.0, 1.5, 0.0]])

    din = np.array([[0, 0.5, 1.0, 1.5, -2.0], [2.0, 0.5, -1.0, 1.5, 0.0]])  # NxM = 2x5

    y, _ = softmax.backward(x=x, din=din)  # NxM = 2x5

    np.testing.assert_allclose(
        y,
        np.array([[0.0, 0.068933, 0.198136, 0.371081, -0.026743], [0.499973, 0.049241, -0.0241, 0.315622, 0.0]]),
        atol=1e-5,
    )


def test_簡単なニューラルネットワークの数値検証():
    # N: 2, d: 5, M: 2
    # 準備
    net = MyNeuralNet(
        layers=[
            Dense(W=np.array([[-2, -1], [-1, -5], [0, 3], [1, -1], [2, -4]]), b=np.array([-1, 2])),
            ReLU(),
        ],
        d_input=5,
        d_output=2,
    )
    x = np.array([[0, 0.5, 1.0, 1.5, 2.0], [2.0, 0.5, 1.0, 1.5, 3.0]])

    # 実行
    y, us = net.forward(x=x)  # NxM=2x2

    # 検証
    np.testing.assert_allclose(y, np.array([[4.0, 0.0], [2.0, 0.0]]))
    np.testing.assert_allclose(us[0], np.array([[0, 0.5, 1.0, 1.5, 2.0], [2.0, 0.5, 1.0, 1.5, 3.0]]))
    np.testing.assert_allclose(us[1], np.array([[4.0, -7.0], [2.0, -13.0]]))


def test_簡単なニューラルネットワークの勾配検証():
    # N: 2, d: 5, M: 2
    # 準備
    net = MyNeuralNet(
        layers=[
            Dense(W=np.array([[-2, -1], [-1, -5], [0, 3], [1, -1], [2, -4]]), b=np.array([-1, 2])),
            ReLU(),
        ],
        d_input=5,
        d_output=2,
    )
    x = np.array([[0, 0.5, 1.0, 1.5, 2.0], [2.0, 0.5, 1.0, 1.5, 3.0]])
    din = np.array([[-3.0, 3.0], [10.0, -10.0]])  # NxM = 2x2

    # 実行
    dout, grads = net.gradient(x=x, din=din)

    # 検証
    np.testing.assert_allclose(grads[0]["W"], np.array([[20.0, 0.0], [3.5, 0.0], [7.0, 0.0], [10.5, 0.0], [24.0, 0.0]]))
    np.testing.assert_allclose(grads[0]["b"], np.array([7.0, 0.0]))


def test_二乗誤差関数の評価():

    net = MyNeuralNet(
        layers=[
            Dense(W=np.array([[-2, -1], [-1, -5], [0, 3], [1, -1], [2, -4]]), b=np.array([-1, 2])),
            ReLU(),
            Dense(W=np.array([[-2], [-1]]), b=np.array([0])),
        ],
        d_input=5,
        d_output=1,
    )
    x = np.array([[0, 0.5, 1.0, 1.5, 2.0], [2.0, 0.5, 1.0, 1.5, 3.0]])
    y = np.array([[-3], [4.0]])
    sqloss = SquaredLoss()

    l = sqloss.eval(net=net, X=x, y=y)

    assert l == 22.25


def test_二乗誤差関数の勾配計算():

    net = MyNeuralNet(
        layers=[
            Dense(W=np.array([[-2, -1], [-1, -5], [0, 3], [1, -1], [2, -4]]), b=np.array([-1, 2])),
            ReLU(),
            Dense(W=np.array([[-2], [-1]]), b=np.array([0])),
        ],
        d_input=5,
        d_output=1,
    )
    x = np.array([[0, 0.5, 1.0, 1.5, 2.0], [2.0, 0.5, 1.0, 1.5, 3.0]])
    y = np.array([[-3], [4.0]])
    sqloss = SquaredLoss()

    grads = sqloss.gradient(net=net, X=x, y=y)

    np.testing.assert_allclose(
        grads[0]["W"], np.array([[16.0, 0.0], [6.5, 0.0], [13.0, 0.0], [19.5, 0.0], [34.0, 0.0]])
    )
    np.testing.assert_allclose(grads[0]["b"], np.array([13.0, 0.0]))
    np.testing.assert_allclose(grads[2]["W"], np.array([[-18.0], [0.0]]))
    np.testing.assert_allclose(grads[2]["b"], np.array([-6.5]))

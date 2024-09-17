# %%
import numpy as np


# 行ごとにソフトマックスを適用する関数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # オーバーフロー対策
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# 数値ヤコビ行列 (中心差分法)
def numerical_jacobian(f, x, epsilon=1e-5):
    N, d = x.shape
    jacobian = np.zeros((N, d, d))

    for i in range(d):  # 各入力要素に対して数値微分を計算
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[:, i] += epsilon
        x_minus[:, i] -= epsilon
        # 各行にソフトマックスを適用した結果の差分を取る
        jacobian[:, :, i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)

    return jacobian


# Nxdの入力行列
x = np.array([[1.0, 2.0, 3.0], [-0.05, 0.0, 0.05]])

# ソフトマックス関数の出力
y = softmax(x)
print("Softmax Output:\n", y)

# ソフトマックス関数の数値ヤコビ行列（微分行列）の計算
jacobian = numerical_jacobian(softmax, x)
print("Numerical Jacobian of Softmax:\n", jacobian)

# %%
# doutを想定する
# Nxdの入力行列
dout = np.array([[1.0, 2.0, 3.0], [-0.05, 0.0, 0.05]])
N, d = x.shape
din = np.zeros_like(dout)
for n in range(N):
    dout_n = dout[n, :]  # 1 x d
    jac_n = jacobian[n, :, :]  # d x d

    din[n, :] = np.dot(dout_n, jac_n)
print(din)

# %%

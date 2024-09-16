# %%
import numpy as np
import pandas as pd

from mydgms.neuralnet import MyNeuralNet, Dense, ReLU, SquaredLoss
from mydgms.training import train_mgd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# %%
#  MNISTデータセットの読み込み
mnist = fetch_openml("mnist_784")

# 特徴量とラベルに分ける
X, y = mnist["data"], mnist["target"]

X.to_csv("./input/mnist_data.csv")
y.to_csv("./input/mnist_target.csv")

# %%
X = pd.read_csv("./input/mnist_data.csv", index_col=0)
y = pd.read_csv("./input/mnist_target.csv", index_col=0).squeeze()

# データの正規化 (0-255のピクセル値を0-1の範囲に)
X = X.astype("float32") / 255.0

# ラベルを数値に変換
y = y.astype("int")

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ラベルをワンホットエンコーディングに変換
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# データをニューラルネットワークの入力形式にリシェイプ (28x28の画像)
# X_train = X_train.reshape(-1, 28, 28, 1)
# X_test = X_test.reshape(-1, 28, 28, 1)

init_net = MyNeuralNet(
    layers=[
        Dense.init(d=784, M=30),
        ReLU(),
        Dense.init(d=30, M=10),
    ],
    d_input=784,
    d_output=10,
)
loss = SquaredLoss()

# 実行
trained_net: MyNeuralNet = train_mgd(
    init_net=init_net,
    loss=loss,
    X=X_train.values,
    y=y_train,
    n_epochs=1000,
    batch_size=20,
    learning_rate=1.0,
)

test_loss = loss.eval(trained_net, X=X_test.values, y=y_test)
assert test_loss

# %%

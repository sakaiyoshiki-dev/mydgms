# %%
import numpy as np
import pandas as pd

from mydgms.neuralnet import MyNeuralNet, Dense, ReLU, Softmax, SquaredLoss
from mydgms.training import train_mgd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# %%
#  MNISTデータセットの読み込み
# mnist = fetch_openml("mnist_784")

# # 特徴量とラベルに分ける
# X, y = mnist["data"], mnist["target"]

# X.to_csv("./input/mnist_data.csv")
# y.to_csv("./input/mnist_target.csv")

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

# %%
# 学習
init_net = MyNeuralNet(
    layers=[
        Dense.init(d=784, M=30, rand=True),
        ReLU(),
        Dense.init(d=30, M=10, rand=True),
        Softmax(),
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
    n_epochs=100,
    batch_size=100,
    learning_rate=0.05,
)

# うまく学習できることもあるが。
# 全然学習が始まらなかったり、途中で急に悪化することがある。
# 学習率が一番の肝、その次にbatch_sizeか？
# また numpy の OPENBLAS_NUM_THREADS 変数も影響していそう。

# %%
# 検証
np.set_printoptions(precision=3, suppress=True)
y_pred, _ = trained_net.forward(X_test.values)
for y_p, y_t in zip(y_pred, y_test):
    print((y_p * y_t).sum())
# test_loss = loss.eval(trained_net, X=X_test.values, y=y_test)

# %%

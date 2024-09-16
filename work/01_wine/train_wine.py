# %%
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # numpyに許可するCPUコア数を制限
import numpy as np
import pandas as pd

from mydgms.neuralnet import MyNeuralNet, Dense, ReLU, Softmax, SquaredLoss
from mydgms.training import train_mgd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# %%
# wineデータセットの読み込み
# wine = fetch_openml("wine")

# # 特徴量とラベルに分ける
# X, y = wine["data"], wine["target"]

# X.to_csv("./input/wine_data.csv")
# y.to_csv("./input/wine_target.csv")

# %%
X = pd.read_csv("./input/wine_data.csv", index_col=0)
y = pd.read_csv("./input/wine_target.csv", index_col=0).squeeze()

# ラベルを数値に変換
y = y.astype("int")

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ラベルをワンホットエンコーディングに変換
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# %%
# 学習
init_net = MyNeuralNet(
    layers=[
        Dense.init(d=13, M=5, rand=True),
        ReLU(),
        Dense.init(d=5, M=2, rand=True),
        ReLU(),
        Dense.init(d=2, M=3, rand=True),
        Softmax(),
    ],
    d_input=13,
    d_output=3,
)
loss = SquaredLoss()

# 実行
trained_net: MyNeuralNet = train_mgd(
    init_net=init_net,
    loss=loss,
    X=X_train.values,
    y=y_train,
    n_epochs=1000,
    batch_size=3,
    learning_rate=0.01,
    X_test=X_test.values,
    y_test=y_test,
)

# うまく学習できることもあるが。
# 全然学習が始まらなかったり、途中で急に悪化することがある。
# 学習率が一番の肝、その次にbatch_sizeか？
# また numpy の OPENBLAS_NUM_THREADS 変数も影響していそう。

# %%
# 検証
from sklearn.metrics import roc_auc_score

y_pred, _ = trained_net.forward(X_test.values)
print(roc_auc_score(y_score=y_pred, y_true=y_test))

# %%

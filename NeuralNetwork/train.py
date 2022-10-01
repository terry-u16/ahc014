# %%
import math
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from network import NeuralNetwork
from parameters import TIMESTAMP
import loader

# %%
TRAINING_DATA_RATIO = 0.75
BATCH_SIZE = 1024

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

# %%
df = loader.load_df()

# %%


def normalize_n(x: pd.DataFrame) -> pd.DataFrame:
    return (x - 31) / 30


def normalize_density(x: pd.DataFrame) -> pd.DataFrame:
    return x * 10


def normalize_temp(x: pd.DataFrame) -> pd.DataFrame:
    return np.log10(x)


def normalize_score(x: pd.DataFrame) -> pd.DataFrame:
    return x / 1e6


# %%
df["N"] = normalize_n(df["N"])
df["Density"] = normalize_density(df["Density"])
df["temp0"] = normalize_temp(df["temp0"])
df["temp1"] = normalize_temp(df["temp1"])
df["score"] = normalize_score(df["score"])
print(df)
# %%
tranining_len = int(len(df) * TRAINING_DATA_RATIO)
df_train = df[:tranining_len]
df_valid = df[tranining_len:]
df_train
# %%
X_COLS = ["N", "Density", "temp0", "temp1"]
Y_COLS = ["score"]
df_X_train = df_train[X_COLS]
df_y_train = df_train[Y_COLS]
df_X_valid = df_valid[X_COLS]
df_y_valid = df_valid[Y_COLS]

# %%
X_train = torch.from_numpy(df_X_train.to_numpy()).float()
y_train = torch.from_numpy(df_y_train.to_numpy()).float()
X_valid = torch.from_numpy(df_X_valid.to_numpy()).float()
y_valid = torch.from_numpy(df_y_valid.to_numpy()).float()

# %%
dataset_train = TensorDataset(X_train, y_train)
dataset_valid = TensorDataset(X_valid, y_valid)

loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE)
# %%
model = NeuralNetwork(len(X_COLS), len(Y_COLS))
model = model.to(device)
# %%
LEARNING_RATE = 0.01
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# %%


class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1e-6), torch.log(actual + 1e-6))


criterion = nn.MSELoss()

# %%


def train_step(train_X: torch.Tensor, train_y: torch.Tensor) -> float:
    model.train()

    train_X = train_X.to(device)
    train_y = train_y.to(device)
    pred_y = model(train_X)
    loss = criterion(pred_y, train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # item()=Pythonの数値


def valid_step(valid_X: torch.Tensor, valid_y: torch.Tensor) -> float:
    model.eval()

    valid_X = valid_X.to(device)
    valid_y = valid_y.to(device)
    pred_y = model(valid_X)  # 出力結果
    loss = criterion(pred_y, valid_y)

    # 損失と正解数をタプルで返す
    return loss.item()
# %%


def init_parameters(layer: nn.Linear):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)  # 重みを「一様分布のランダム値」に初期化
        layer.bias.data.fill_(0.0)            # バイアスを「0」に初期化


# %%
# 学習の前にパラメーター（重みやバイアス）を初期化する
model.apply(init_parameters)

# 定数（学習／評価時に必要となるもの）
EPOCHS = 20

# 変数（学習／評価時に必要となるもの）
avg_loss = 0.0           # 「訓練」用の平均「損失値」
avg_acc = 0.0            # 「訓練」用の平均「正解率」
avg_val_loss = 0.0       # 「評価」用の平均「損失値」
avg_val_acc = 0.0        # 「評価」用の平均「正解率」

# 損失の履歴を保存するための変数
train_history = []
valid_history = []

for epoch in range(EPOCHS):
    # forループ内で使う変数と、エポックごとの値リセット
    total_loss = 0.0     # 「訓練」時における累計「損失値」
    total_val_loss = 0.0  # 「評価」時における累計「損失値」
    total_train = 0      # 「訓練」時における累計「データ数」
    total_valid = 0      # 「評価」時における累計「データ数」

    for train_X, train_y in loader_train:
        loss = train_step(train_X, train_y)

        total_loss += loss
        total_train += len(train_y)

    for valid_X, valid_y in loader_valid:
        val_loss = valid_step(valid_X, valid_y)

        total_val_loss += val_loss
        total_valid += len(valid_y)

    avg_loss = total_loss / total_train
    avg_val_loss = total_val_loss / total_valid

    # グラフ描画のために損失の履歴を保存する
    train_history.append(avg_loss)
    valid_history.append(avg_val_loss)

    # 損失や正解率などの情報を表示
    print(f'[Epoch {epoch+1:4d}/{EPOCHS:3d}] loss: {math.sqrt(avg_loss):.6f} val_loss: {math.sqrt(avg_val_loss):.6f}')

print('Finished Training')
print(model.state_dict())  # 学習後のパラメーターの情報を表示
# %%
current_time = datetime.datetime.now()
model_path = f"./data/models/{TIMESTAMP}.pth"
torch.save(model.cpu().state_dict(), model_path)
# %%
n = X_train[0].numpy()[0]
density = X_train[0].numpy()[1]
print(f"{n} {density}")

for i in range(11):
    for j in range(11):
        temp0 = np.log10(5 * np.power(10.0, i / 10))
        temp1 = np.log10(np.power(10.0, j / 10))

        print(f"{temp0} {temp1}")
        x = torch.from_numpy(
            np.array([n, density, temp0, temp1])).float().to(device)
        score = model.to(device)(x).cpu().detach().numpy()
        print(f"{np.power(10, temp0):.2f} {np.power(10, temp1):.2f} {score*1000000}")

# %%

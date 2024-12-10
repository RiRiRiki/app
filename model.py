import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# PyTorchモデルの定義
class StockPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(StockPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# データの前処理とトレーニング関数
def prepare_data(df):
    df_rate = pd.DataFrame()
    df_rate['TOPIX 1 day return'] = df['1306.T'].pct_change()
    df_rate['7203 1 day return'] = df['7203.T'].pct_change()
    df_rate = df_rate.dropna()
    df_rate['diff'] = df_rate['7203 1 day return'].shift(-1) - df_rate['TOPIX 1 day return'].shift(-1)
    df_rate['target'] = (df_rate['diff'] > 0).astype(int)
    df_rate = df_rate.dropna()

    diffs = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for diff in diffs:
        df_rate[f'TOPIX {diff} days return'] = df['1306.T'].pct_change(diff)
        df_rate[f'7203 {diff} days return'] = df['7203.T'].pct_change(diff)

    df_rate = df_rate.dropna(how='any')
    X = df_rate[df_rate.columns.difference(['diff', 'target'])]
    y = df_rate['target']

    return train_test_split(X, y, test_size=0.2, shuffle=False)

# モデルのトレーニング
def train_model(X_train, y_train, input_size):
    model = StockPredictionModel(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()
    return model

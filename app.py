import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf

target = '7203.T'
symbols = ('1306.T',target)
df = pd.DataFrame()
for symbol in symbols:
    data = yf.download(symbol,
                              start="2016-03-30",
                              end="2021-03-31")['Adj Close']
    data.name = symbol
    df = pd.concat([df,data],axis = 1)

for i in np.arange(len(df.index)):
    df.index.values[i] = str(df.index[i].date())

df.index = df.index.rename('Date')

df = df.dropna()

df_rate = pd.DataFrame()
df_rate['TOPIX 1 day return'] = df['1306.T'].pct_change()
df_rate['7203 1 day return'] = df['7203.T'].pct_change()
df_rate = df_rate.dropna()
df_rate['diff'] =  df_rate['7203 1 day return'].shift(-1)\
                - df_rate['TOPIX 1 day return'].shift(-1)
df_rate['target'] = (df_rate['diff'] > 0).astype(int)
df_rate = df_rate.dropna()

diffs = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,40,60,80,100,120,\
         140,160,180,200,220,240]

for diff in diffs:
    df_rate['TOPIX ' + str(diff) + ' days return'] = df['1306.T'].pct_change(diff)
    df_rate['7203 ' + str(diff) + ' days return'] = df['7203.T'].pct_change(diff)

df_rate = df_rate.dropna(how = 'any')

X = df_rate[df_rate.columns[(df_rate.columns != 'diff')
                            & (df_rate.columns != 'target')]]
y = df_rate['target']

from sklearn.model_selection import train_test_split

val_size = 0.2

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = val_size,shuffle = False)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

model = Sequential()
for _ in range(10):
    model.add(Dense(62,activation = 'selu',
                    kernel_initializer = 'lecun_normal'))
    model.add(Dropout(0.5))
model.add(Dense(1,activation = 'sigmoid'))
optimizer = optimizers.RMSprop(learning_rate = 0.001,rho=0.99)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy',
             metrics = ['accuracy'])
history_drop2 = model.fit(X_train.values,y_train.values,
         epochs = 100, batch_size = 10,verbose = 2,
         validation_data = (X_val.values,y_val.values))

score_train = model.evaluate(X_train.values,y_train.values,verbose = 0)
score_val = model.evaluate(X_val.values,y_val.values,verbose = 0)
print('accuracy for train data',score_train[1])
print('accuracy for validation data',score_val[1])

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model

# モデルをロード
model = load_model('model.h5')  # 事前に保存されたモデルをロード

# タイトル
st.title("株価予測アプリ")

# 入力フォーム
target = st.text_input("ターゲット銘柄（例: '7203.T'）", value="7203.T")
start_date = st.date_input("開始日", value=pd.Timestamp("2016-01-01"))
end_date = st.date_input("終了日", value=pd.Timestamp("2021-12-31"))

# 実行ボタン
if st.button("予測を実行"):
    # データを取得
    yf.pdr_override()
    symbols = ["1306.T", target]
    df = pd.DataFrame()

    for symbol in symbols:
        data = yf.download(symbol, start=start_date, end=end_date)["Adj Close"]
        data.name = symbol
        df = pd.concat([df, data], axis=1)

    # データの前処理
    df_rate = pd.DataFrame()
    df_rate['TOPIX 1 day return'] = df['1306.T'].pct_change()
    df_rate[f'{target} 1 day return'] = df[target].pct_change()
    df_rate['diff'] = df_rate[f'{target} 1 day return'].shift(-1) - df_rate['TOPIX 1 day return'].shift(-1)
    df_rate['target'] = (df_rate['diff'] > 0).astype(int)
    df_rate = df_rate.dropna()

    diffs = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for diff in diffs:
        df_rate[f'TOPIX {diff} days return'] = df['1306.T'].pct_change(diff)
        df_rate[f'{target} {diff} days return'] = df[target].pct_change(diff)

    df_rate = df_rate.dropna(how='any')
    X = df_rate.drop(columns=['diff', 'target'])
    y = df_rate['target']

    # モデルで予測
    predictions = model.predict(X.values)

    # 最新の日付で結果を表示
    latest_date = df_rate.index[-1]
    latest_prediction = predictions[-1][0]
    result = "ターゲット銘柄が上回る" if latest_prediction > 0.5 else "TOPIXが上回る"
    st.write(f"{latest_date} 時点での予測: {result} (確率: {latest_prediction:.2%})")

    # 期間全体の予測結果を表示
    df_rate['prediction'] = predictions
    st.write("期間全体の予測結果:")
    st.dataframe(df_rate[['prediction']])

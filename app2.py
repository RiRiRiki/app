import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

# タイトル
st.title("株価予測アプリ")

# ファイル確認
MODEL_FILE = "model.h5"

# モデルをトレーニングして保存
def train_and_save_model(X_train, y_train, input_dim):
    model = Sequential()
    for _ in range(10):
        model.add(Dense(62, activation='selu', kernel_initializer='lecun_normal'))
        model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = RMSprop(learning_rate=0.001, rho=0.99)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # モデル学習
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=2)
    model.save(MODEL_FILE)
    return model

# 入力フォーム
target = st.text_input("ターゲット銘柄（例: '7203.T'）", value="7203.T")
start_date = st.date_input("開始日", value=pd.Timestamp("2016-01-01"))
end_date = st.date_input("終了日", value=pd.Timestamp("2021-12-31"))

# データ取得と前処理
if st.button("予測を実行"):
    # データ取得（直接yfinanceを使用）
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
    X = df_rate.drop(columns=['diff', 'target']).values
    y = df_rate['target'].values

    # データ分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # モデルロードまたはトレーニング
    if os.path.exists(MODEL_FILE):
        st.write("既存のモデルをロードしています...")
        model = load_model(MODEL_FILE)
    else:
        st.write("モデルをトレーニングしています...")
        model = train_and_save_model(X_train, y_train, X_train.shape[1])

    # 予測
    predictions = model.predict(X_val)
    latest_prediction = predictions[-1][0]
    result = "ターゲット銘柄が上回る" if latest_prediction > 0.5 else "TOPIXが上回る"
    st.write(f"最新の予測結果: {result} (確率: {latest_prediction:.2%})")

    # 結果の可視化
    df_rate['prediction'] = model.predict(X).flatten()
    st.write("予測結果（全期間）:")
    st.dataframe(df_rate[['prediction']])

    # グラフ表示
    st.line_chart(df_rate['prediction'])

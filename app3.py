import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# タイトル
st.title("株価予測アプリ")

# ファイル確認
MODEL_FILE = "model.h5"

# モデルをトレーニングして保存
def train_and_save_model(X_train, y_train):
    input_dim = X_train.shape[1]  # 特徴量数に基づいて動的に設定
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # コールバック設定
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    # モデル学習
    model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=64,
        verbose=2,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr]
    )
    model.save(MODEL_FILE)
    return model

# 入力フォーム
target = st.text_input("ターゲット銘柄（例: '7203.T'）", value="7203.T")
start_date = st.date_input("開始日", value=pd.Timestamp("2013-01-01"))
end_date = st.date_input("終了日", value=pd.Timestamp("2024-12-01"))

# データ取得と前処理
if st.button("予測を実行"):
    # データ取得
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

    diffs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    for diff in diffs:
        df_rate[f'TOPIX {diff} days return'] = df['1306.T'].pct_change(diff)
        df_rate[f'{target} {diff} days return'] = df[target].pct_change(diff)

    df_rate = df_rate.dropna(how='any')
    X = df_rate.drop(columns=['diff', 'target']).values
    y = df_rate['target'].values

    # 特徴量の標準化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # データ分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

    # 入力データの形状確認
    st.write(f"X_train shape: {X_train.shape}")

    # モデルロードまたはトレーニング
    if os.path.exists(MODEL_FILE):
        st.write("既存のモデルをロードしています...")
        model = load_model(MODEL_FILE)

        # 保存済みモデルの入力次元が異なる場合、再トレーニング
        if model.input_shape[1] != X_train.shape[1]:
            st.write("入力次元が異なるため、モデルを再トレーニングします...")
            model = train_and_save_model(X_train, y_train)
    else:
        st.write("モデルをトレーニングしています...")
        model = train_and_save_model(X_train, y_train)

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

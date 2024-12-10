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

# ボラティリティの計算
def calculate_volatility(data, window=14):
    return data.pct_change().rolling(window=window).std()

# タイトル
st.title("株価予測アプリ - ボラティリティと出来高対応版\n(製作中のため精度低いです)")

# ファイル確認
MODEL_FILE = "model_with_volatility_volume.h5"

# モデルをトレーニングして保存
def train_and_save_model(X_train, y_train):
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # コールバック設定
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=64,
        verbose=2,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr]
    )
    model.save(MODEL_FILE)
    return model

# 入力フォーム
target = st.text_input("ターゲット銘柄（例: '7203.T'）", value="7203.T")
start_date = st.date_input("開始日", value=pd.Timestamp("2014-01-01"))
end_date = st.date_input("終了日", value=pd.Timestamp("2024-12-31"))

# データ取得と前処理
if st.button("予測を実行"):
    # データ取得
    symbols = ["1306.T", target]
    df = pd.DataFrame()

    for symbol in symbols:
        data = yf.download(symbol, start=start_date, end=end_date)[["Adj Close", "Volume"]]
        data.columns = [f"{symbol}_close", f"{symbol}_volume"]
        df = pd.concat([df, data], axis=1)

    # 特徴量計算
    df['target_volatility'] = calculate_volatility(df[f"{target}_close"], window=14)
    df['TOPIX_volatility'] = calculate_volatility(df["1306.T_close"], window=14)
    df['target_volume'] = df[f"{target}_volume"]
    df['TOPIX_volume'] = df["1306.T_volume"]

    # データの前処理
    df_rate = pd.DataFrame()
    df_rate['TOPIX 1 day return'] = df["1306.T_close"].pct_change()
    df_rate[f'{target} 1 day return'] = df[f"{target}_close"].pct_change()
    df_rate['diff'] = df_rate[f'{target} 1 day return'].shift(-1) - df_rate['TOPIX 1 day return'].shift(-1)
    df_rate['target'] = (df_rate['diff'] > 0).astype(int)

    df_rate['target_volatility'] = df['target_volatility']
    df_rate['TOPIX_volatility'] = df['TOPIX_volatility']
    df_rate['target_volume'] = df['target_volume']
    df_rate['TOPIX_volume'] = df['TOPIX_volume']

    diffs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    for diff in diffs:
        df_rate[f'TOPIX {diff} days return'] = df["1306.T_close"].pct_change(diff)
        df_rate[f'{target} {diff} days return'] = df[f"{target}_close"].pct_change(diff)

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
    model = None  # 初期化して明示的にmodelを定義

    if os.path.exists(MODEL_FILE):
        st.write("既存のモデルをロードしています...")
        model = load_model(MODEL_FILE)

        if model.input_shape[1] != X_train.shape[1]:
            st.write("入力次元が異なるため、モデルを再トレーニングします...")
            model = train_and_save_model(X_train, y_train)
    else:
        st.write("モデルをトレーニングしています...")
        model = train_and_save_model(X_train, y_train)

    # モデルがロードまたは作成されたことを確認
    if model is None:
        st.error("モデルのロードまたはトレーニングに失敗しました。")
    else:
        # 予測部分
        predictions = model.predict(X_val).flatten()
        latest_prediction = predictions[-1]

        # 結果解釈
        if latest_prediction > 0.5:
            result = "ターゲット銘柄が上回る"
            probability = latest_prediction
        else:
            result = "TOPIXが上回る"
            probability = 1 - latest_prediction

        st.write(f"最新の予測結果: {result} (確率: {probability:.2%})")

        # 結果の可視化
        df_rate['prediction'] = model.predict(X).flatten()
        st.write("次の日（指定の期間後）に、ターゲット銘柄のリターンが\nTOPIXのリターンを上回る確率を算出します。")
        st.write("予測結果（全期間）:")
        st.dataframe(df_rate[['prediction']])
        
        # グラフ表示
        st.line_chart(df_rate['prediction'])

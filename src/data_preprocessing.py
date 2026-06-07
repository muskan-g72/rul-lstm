import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

SEQ_LEN = 30

def load_data():
    train_df = pd.read_csv("train_FD001.txt", sep=" ", header=None)
    test_df = pd.read_csv("test_FD001.txt", sep=" ", header=None)

    train_df = train_df.drop(columns=[26, 27])
    test_df = test_df.drop(columns=[26, 27])

    rul_test = pd.read_csv(
        "RUL_FD001.txt",
        header=None,
        names=["true_RUL"]
    )

    columns = ["unit", "cycle"] + \
              [f"setting{i}" for i in range(1, 4)] + \
              [f"sensor{i}" for i in range(1, 22)]

    train_df.columns = columns
    test_df.columns = columns

    return train_df, test_df, rul_test

def preprocess_data(train_df):

    rul_df = train_df.groupby("unit")["cycle"].max().reset_index()
    rul_df.columns = ["unit", "max_cycle"]

    train_df = train_df.merge(rul_df, on="unit")

    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
    train_df["RUL"] = train_df["RUL"].clip(upper=125)

    train_df.drop(columns=["max_cycle"], inplace=True)

    return train_df

def prepare_features(train_df):

    all_units = train_df["unit"].unique()

    np.random.seed(42)

    train_units = np.random.choice(
        all_units,
        size=int(0.8 * len(all_units)),
        replace=False
    )

    val_units = [u for u in all_units if u not in train_units]

    train_data = train_df[train_df["unit"].isin(train_units)]
    val_data = train_df[train_df["unit"].isin(val_units)]

    all_sensor_cols = [f"sensor{i}" for i in range(1, 22)]

    selector = VarianceThreshold(threshold=0.001)

    selector.fit(train_data[all_sensor_cols])

    good_sensors = [
        s for s, keep in zip(
            all_sensor_cols,
            selector.get_support()
        ) if keep
    ]

    good_sensors = [s for s in good_sensors if s != "sensor9"]

    feature_cols = [
        "setting1",
        "setting2",
        "setting3"
    ] + good_sensors

    scaler = MinMaxScaler()

    scaler.fit(train_data[feature_cols])

    return train_data, val_data, scaler, feature_cols

def create_sequences(data, seq_len, feature_cols, scaler):

    X, y = [], []

    for unit in data["unit"].unique():

        unit_df = data[data["unit"] == unit]

        scaled = scaler.transform(unit_df[feature_cols])

        rul = unit_df["RUL"].values

        for i in range(len(unit_df) - seq_len):

            X.append(scaled[i:i+seq_len])
            y.append(rul[i+seq_len])

    return np.array(X), np.array(y)

def create_test_sequences(data, seq_len, feature_cols, scaler):

    X = []

    for unit in sorted(data["unit"].unique()):

        unit_df = data[data["unit"] == unit]

        scaled = scaler.transform(unit_df[feature_cols])

        if len(unit_df) >= seq_len:

            X.append(scaled[-seq_len:])

        else:

            pad = np.zeros(
                (seq_len-len(unit_df), len(feature_cols))
            )

            X.append(np.vstack([pad, scaled]))

    return np.array(X)
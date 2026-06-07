from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from data_preprocessing import *
from model import build_lstm_model
from evaluate import evaluate_model

train_df, test_df, rul_test = load_data()

train_df = preprocess_data(train_df)

train_data, val_data, scaler, feature_cols = \
    prepare_features(train_df)

X_train, y_train = create_sequences(
    train_data,
    SEQ_LEN,
    feature_cols,
    scaler
)

X_val, y_val = create_sequences(
    val_data,
    SEQ_LEN,
    feature_cols,
    scaler
)

X_test = create_test_sequences(
    test_df,
    SEQ_LEN,
    feature_cols,
    scaler
)

y_test = rul_test["true_RUL"].values.clip(max=125)

model = build_lstm_model(
    (SEQ_LEN, X_train.shape[2])
)

callbacks = [
    EarlyStopping(
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        factor=0.5,
        patience=5
    )
]

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

y_pred = model.predict(X_test)

evaluate_model(y_test, y_pred)
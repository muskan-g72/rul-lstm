import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def nasa_loss(y_true, y_pred):

    diff = y_pred - y_true

    score = tf.reduce_mean(
        tf.where(
            diff < 0,
            tf.exp(-diff/13)-1,
            tf.exp(diff/10)-1
        )
    )

    return score

def build_lstm_model(input_shape):

    model = Sequential()

    model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=input_shape
        )
    )

    model.add(Dropout(0.2))

    model.add(LSTM(32))

    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss=nasa_loss
    )

    return model
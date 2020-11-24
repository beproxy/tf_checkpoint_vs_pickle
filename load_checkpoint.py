import os
from time import perf_counter
from datetime import datetime
import logging
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import tensorflow as tf
from machinery import Machinery


def create_model():
    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE,
                   activation='tanh',
                   recurrent_activation='tanh',
                   return_sequences=True,
                   use_bias=True,
                   input_shape=(60, 1),
                   unroll=True))
    model.add(Dropout(0.25))
    model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(HIDDEN_SIZE))
    model.add(Dropout(0.25))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer="rmsprop")
    return model


if __name__ == '__main__':
    dt = str(datetime.now())
    logging.basicConfig(filename='log.log', level=logging.INFO)
    logging.info(f'----- Starting process at {dt}')

    # SETUP
    SEQLEN = 60  # Sequence length
    HIDDEN_SIZE = 128
    BATCH_SIZE = 128
    EPOCHS_ITERATIONS = 5

    INPUT_FILE = "yahoo_finance.csv"

    machinery = Machinery(INPUT_FILE, SEQLEN)
    logging.info("dataset extracted and split, train data and test data")

    # Train set is split between X and y and reshape
    X_train, y_train = machinery.x_y_normalize
    logging.info("Train set is split between X and y and reshape")

    # Create model
    model = create_model()

    # Make prediction without loading weights
    predicted_data = machinery.prediction_data(model)

    # Evaluate model
    eval_model = machinery.return_rmse(predicted_data)
    logging.info(f'Evaluate model without training: {eval_model} (mean squared error)')
    print(f'Evaluate model without training: {eval_model} (mean squared error)')

    # Timer
    t1_start = perf_counter()

    # load weights
    checkpoint_path = "training_ckpt/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    latest = tf.train.latest_checkpoint(checkpoint_dir)

    # Create model
    model = create_model()

    # Set weights in model
    model.load_weights(latest)

    t1_stop = perf_counter()
    timer = t1_stop - t1_start
    print(f'Time to loading Checkpoint {timer} sec.')
    logging.info(f'Time to loading Checkpoint {timer} sec.')

    # If you want again train model
    # model.fit(...)

    # Make prediction with loading weights
    predicted_data = machinery.prediction_data(model)

    # Evaluate model
    eval_model = machinery.return_rmse(predicted_data)
    logging.info(f'Evaluate model with set weights: {eval_model} (mean squared error)')
    print(f'Evaluate model with set weights: {eval_model} (mean squared error)')
    logging.info('----- End of process')

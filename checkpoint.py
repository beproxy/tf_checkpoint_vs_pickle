import os
from datetime import datetime
import logging
from time import perf_counter

from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from machinery import Machinery


def create_model():
    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE,
                   activation='tanh',
                   recurrent_activation='tanh',
                   return_sequences=True,
                   use_bias=True,
                   input_shape=(X_train.shape[1], 1),
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


# func for visualizing of data
def visual_data(data_1, data_2, title, label_1, label_2, x_label, y_label, filename):
    plt.plot(data_1, label=label_1)
    plt.plot(data_2, label=label_2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    dt = str(datetime.now())
    logging.basicConfig(filename='log.log', level=logging.INFO)
    logging.info(f'----- Starting process at {dt}')

    # SETUP
    SEQLEN = 60  # Sequence length
    HIDDEN_SIZE = 128
    BATCH_SIZE = 128
    EPOCHS_ITERATIONS = 5
    PLOT_NUMBER = '2'  # adding to plots file a number
    logging.info(f'Plot number: {PLOT_NUMBER}. Epochs {EPOCHS_ITERATIONS}. Model LSTM with mutable hyperparams:' +
                 f' hidden size={HIDDEN_SIZE}, activation=tanh, recurrent_activation=tanh' +
                 '\nLayers: LSTM / Dropout(0.25) 4th, Dense=1')

    # Open file
    INPUT_FILE = "yahoo_finance.csv"
    machinery = Machinery(INPUT_FILE, SEQLEN)
    logging.info("dataset extracted and split, train data and test data")

    # Train set is split between X and y and reshape
    X_train, y_train = machinery.x_y_normalize
    logging.info("Train set is split between X and y and reshape")

    # Model
    model = create_model()
    model_summary = model.summary()

    # Timer
    t1_start = perf_counter()

    # create checkpoint
    checkpoint_path = "training_ckpt/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  verbose=1,
                                  period=1)  # PERIOD - saving weights every 5 epochs

    # Save weights
    model.save_weights(checkpoint_path.format(epoch=0))

    # Training model
    history = model.fit(X_train,
                        y_train,
                        epochs=EPOCHS_ITERATIONS,
                        batch_size=BATCH_SIZE,
                        callbacks=[cp_callback])  # Pass callback to training
    print(f'Loss: {history.history["loss"]}')
    logging.info(f'Loss: {history.history["loss"]}')

    # Timer result
    t1_stop = perf_counter()
    timer = t1_stop - t1_start
    print(f'Save weight through Checkpoint: {timer} sec.')
    logging.info(f'Save weight through Checkpoint: {timer} sec.')

    # Visualization Loss
    plt.plot(history.history["loss"], color='red')
    plt.title('Visualization loss data (LSTM model)')
    plt.savefig(f'visual_loss_{PLOT_NUMBER}.png')
    plt.show()

    # visualization of data set with show training data set and testing data set
    if PLOT_NUMBER == False:
        data_plot = [machinery.dataset["High"][:'2019'],
                     machinery.dataset["High"]['2019':],
                     'Yahoo Finance High stock price',
                     'Training set (Before 2019)',
                     'Testing set (After 2019)',
                     'Date',
                     'High Price',
                     'dataset_plot.png'
                     ]
        visual_data(*data_plot)

    # Prediction
    predicted_data = machinery.prediction_data(model)

    # visualization LSTM model of test set and predicted data
    data_plot = [predicted_data,
                 machinery.test_set,
                 'Yahoo Finance High Price Prediction',
                 'Predicted High Price',
                 'Real High Price',
                 'Date',
                 'High Price',
                 f'prediction_plot_{PLOT_NUMBER}.png'
                 ]
    visual_data(*data_plot)

    # Evaluate model
    eval_model = machinery.return_rmse(predicted_data)
    print(f'Evaluate model: {eval_model}')
    logging.info(f'Evaluate model: {eval_model}')
    logging.info('----- End of process')


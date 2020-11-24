from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error


class Machinery:
    def __init__(self, filename, seqlen):
        self.seqlen = seqlen
        self.filename = filename
        self.dataset = self.open_dataset()
        self.train_set, self.test_set = self.split_train_test_data()
        self.sc = MinMaxScaler(feature_range=(0, 1))
        self.train_set_scaled = self.trainset_fit_transform()
        self.x_y_normalize = self.split_train_x_y()
        self.X_test = self.testset_transform()

    # Extracting data from file
    def open_dataset(self):
        print(f"Extracting data from file {self.filename}...")
        dataset = pd.read_csv(self.filename)
        dataset.set_index('Date', inplace=True)
        return dataset

    # Data set is split between training set and testing set
    def split_train_test_data(self):
        train_set = self.dataset[:'2019'].iloc[:, 1:2].values
        test_set = self.dataset['2019':].iloc[:, 1:2].values
        return train_set, test_set

    # Minimize training data set
    def trainset_fit_transform(self):
        return self.sc.fit_transform(self.train_set)

    # Train set is split between X and y and reshape
    def split_train_x_y(self):
        print("Train set is split between X and y...")
        X, y = [], []
        for i in range(self.seqlen, self.train_set.shape[0]):
            X.append(self.train_set_scaled[i - self.seqlen:i, 0])
            y.append(self.train_set_scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    # Transform testing dataset and reshape
    def testset_transform(self):
        dataset_total = pd.concat((self.dataset["High"][:'2018'], self.dataset["High"]['2019':]), axis=0)
        inputs = dataset_total[len(dataset_total) - len(self.test_set) - self.seqlen:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.sc.transform(inputs)
        X_test = []
        for i in range(self.seqlen, inputs.shape[0]):
            X_test.append(inputs[i - self.seqlen:i, 0])
        X_test = np.array(X_test)
        return np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Prediction
    def prediction_data(self, model):
        prediction = model.predict(self.X_test)
        return self.sc.inverse_transform(prediction)

    # Evaluate model (root mean square error)
    def return_rmse(self, prediction):
        return math.sqrt(mean_squared_error(self.test_set, prediction))

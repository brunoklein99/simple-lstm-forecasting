import matplotlib.pyplot as plt
import numpy as np
from export import export_model
from data import prepare_and_save_data
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import SGD


def smoothing(x, beta=0.9):
    x = np.copy(x)
    v = x[0]
    for i in range(len(x) - 2):
        v = beta * v + (1 - beta) * x[i]
        x[i] = v
    return x


def window_transform_series(series, window_size):
    X = []
    y = []

    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


if __name__ == '__main__':
    prepare_and_save_data()
    original_data = np.load('data/invoices.npy')
    original_data = np.array(original_data, dtype=np.float64)
    original_data /= max(original_data)
    original_data = original_data * 2
    original_data = original_data - 1
    smoothed_data = smoothing(original_data)

    plt.plot(original_data, 'r')
    plt.plot(smoothed_data, 'g')
    plt.xlabel('time period')
    plt.ylabel('normalized series value')
    plt.show()

    window_size = 7
    X, y = window_transform_series(series=smoothed_data, window_size=window_size)

    train_test_split = int(np.ceil(2 * len(y) / float(3)))  # set the split point

    X_train = X[:train_test_split, :]
    y_train = y[:train_test_split]

    X_test = X[train_test_split:, :]
    y_test = y[train_test_split:]

    X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
    X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))

    np.random.seed(0)

    model = Sequential()
    model.add(LSTM(100, input_shape=(window_size, 1)))
    model.add(Dense(1))

    optimizer = SGD(lr=0.01, momentum=0.9)

    model.compile(loss='mae', optimizer=optimizer)

    training_info = model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)

    loss = training_info.history['loss']

    plt.plot(loss)
    plt.show()

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    training_error = model.evaluate(X_train, y_train, verbose=0)
    print('training error = ' + str(training_error))

    testing_error = model.evaluate(X_test, y_test, verbose=0)
    print('testing error = ' + str(testing_error))

    plt.plot(original_data, color='k')
    plt.plot(smoothed_data, 'g')

    split_pt = train_test_split + window_size
    plt.plot(np.arange(window_size, split_pt, 1), train_predict, color='b')

    plt.plot(np.arange(split_pt, split_pt + len(test_predict), 1), test_predict, color='r')

    plt.xlabel('day')
    plt.ylabel('invoices')
    plt.legend([
        'original series',
        'smoothed data',
        'training fit',
        'testing fit'
    ], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    export_model(model)

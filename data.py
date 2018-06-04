from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def start_of_week(x: datetime):
    x = x - timedelta(days=x.today().isoweekday() % 7)
    x = x.date()
    return x


def end_of_week(x):
    return start_of_week(x) + timedelta(days=7)


def prepare_and_save_data():
    _, data = load_data()
    data = np.array(data)
    np.save('data/invoices.npy', data)


def load_data():
    column = 'Inovice Date'

    df = pd.read_csv('data/invoices.csv')
    df = pd.to_datetime(df[column], infer_datetime_format=True)

    max_date = max(df)
    min_date = min(df)

    max_date = end_of_week(max_date)
    min_date = start_of_week(min_date)

    index = pd.date_range(start=min_date, end=max_date, freq='1D')

    series = pd.Series([0] * len(index), index=index)

    df = df.value_counts()

    for i, (d, v) in enumerate(df.iteritems()):
        series[d] = series[d] + v

    series = series.resample('1D').sum()

    ticks = [str(x.date()) for x in list(series.keys())]

    return ticks, series.values


def window_transform_series(series, window_size):
    # containers for input/output pairs
    x = []
    y = []

    for i in range(window_size, len(series)):
        x.append(series[i - window_size:i])
        y.append(series[i])

    # reshape each
    x = np.asarray(x)
    x.shape = (np.shape(x)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return x, y
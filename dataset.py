import numpy as np


def train_test(df, percentage=0.8):
    return df[:int(len(df) * percentage)], df[int(len(df) * percentage):]


def create_chunks(df, seq_len, n_col=0):
    data = df.reset_index(drop=True).values
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])  # Chunks of training data with a length of seq_len rows
        y.append(data[:, n_col][i])  # Value to predict
    X, y = np.array(X), np.array(y)
    return X, y

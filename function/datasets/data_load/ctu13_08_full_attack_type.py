import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

fileDir = os.path.dirname(os.path.abspath("__file__"))

test_size = 0.2
val_size = 0.125
random_state = 1


def ctu13_08_full_attack_type():
    """Load CTU13_08 keeping multi-class attack labels intact.

    Assumes CSV with features in all columns except last (label). Labels are ints.
    Returns: X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal
    """
    path = os.path.join(fileDir, "./data/ctu13-08/CTU13_08.csv")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)

    # Features and labels
    X = data[:, :-1]
    y = data[:, -1]

    y[y == 1] = 0
    y[y == 2] = 1

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Val split from train
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )  # 0.125 * 0.8 = 0.1 of total

    # Mal pool: all non-zero labels
    X_mal = X_train[y_train != 0]
    y_mal = y_train[y_train != 0]

    return (
        np.asarray(X_train, dtype=np.float64),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(X_val, dtype=np.float64),
        np.asarray(y_val, dtype=np.int64),
        np.asarray(X_test, dtype=np.float64),
        np.asarray(y_test, dtype=np.int64),
        np.asarray(X_mal, dtype=np.float64),
        np.asarray(y_mal, dtype=np.int64),
    )

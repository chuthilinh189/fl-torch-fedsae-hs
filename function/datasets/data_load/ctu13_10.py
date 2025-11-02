import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'
fileDir = os.path.dirname(os.path.abspath("__file__"))

test_size = 0.2
val_size = 0.125
random_state = 1


def ctu13_10():
    pathData = os.path.join(fileDir, "./data/ctu13-10/CTU13_10.csv")
    Data = np.genfromtxt(pathData, delimiter=",")

    print("Data size", Data.shape)

    X = Data[:, 0:-1]
    y = Data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)

    mmsc = MinMaxScaler()
    X_train = mmsc.fit_transform(X_train)
    X_test = mmsc.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )  # 0.125 x 0.8 = 0.1

    X_mal = X_train[y_train > 1]
    y_mal = y_train[y_train > 1]

    X_train = X_train[y_train <= 1]
    y_train = y_train[y_train <= 1]

    print(
        "Final Size: ",
        (X_train.shape, y_train.shape),
        (X_val.shape, y_val.shape),
        (X_test.shape, y_test.shape),
        (X_mal.shape, y_mal.shape),
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

fileDir = os.path.dirname(os.path.abspath("__file__"))

test_size = 0.2
val_size = 0.125
random_state = 1


def cic_ids():
    pathTrain = os.path.join(fileDir, "./data/cic-ids/CIC-IDS-2017-Train-10percent.csv")
    pathTest = os.path.join(fileDir, "./data/cic-ids/CIC-IDS-2017-Test-10percent.csv")

    TrainData = pd.read_csv(pathTrain, sep=",", header=None)
    TestData = pd.read_csv(pathTest, sep=",", header=None)

    TrainData.replace([np.inf, -np.inf], np.nan, inplace=True)
    TrainData.dropna(inplace=True)
    TestData.replace([np.inf, -np.inf], np.nan, inplace=True)
    TestData.dropna(inplace=True)

    Data = pd.concat([TrainData, TestData])
    y = Data[85]
    y.loc[y != "BENIGN"] = 1
    y.loc[y == "BENIGN"] = 0

    droppedColumn = [0, 1, 2, 4, 7, 85]
    X = Data
    for idx in droppedColumn:
        X = X.drop([idx], axis=1)

    print("Original Size: ", (X.shape, y.shape))

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

    print("Splitted Size: ", (X_train.shape, y_train.shape), (X_val.shape, y_val.shape), (X_test.shape, y_test.shape))

    X_mal = X_train[y_train == 1]
    y_mal = y_train[y_train == 1]

    # X_train = X_train[y_train == 0]
    # y_train = y_train[y_train == 0]

    # X_val = X_val[y_val == 0]
    # y_val = y_val[y_val == 0]

    print(
        "Final Size: ",
        (X_train.shape, y_train.shape),
        (X_val.shape, y_val.shape),
        (X_test.shape, y_test.shape),
        (X_mal.shape, y_mal.shape),
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal

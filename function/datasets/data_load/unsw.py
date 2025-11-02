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


def unsw():
    pathTrain = os.path.join(fileDir, "./data/unsw/UNSW_Train.csv")
    pathTest = os.path.join(fileDir, "./data/unsw/UNSW_Test.csv")

    TrainData = pd.read_csv(pathTrain, sep=",", header=None)
    TestData = pd.read_csv(pathTest, sep=",", header=None)

    print("Train size: ", TrainData.shape)
    print("Test size: ", TestData.shape)

    Data = pd.concat([TrainData, TestData])
    # Data = Data.sample(frac=0.1, random_state=random_state)

    y = Data.iloc[:, -1]
    y[y > 0] = 1
    X = Data.iloc[:, 0:-1]

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

    print("Benign val: ", y_val[y_val == 0].shape)
    print("Mal val: ", y_val[y_val == 1].shape)
    print("Benign test: ", y_test[y_test == 0].shape)
    print("Mal test: ", y_test[y_test == 1].shape)

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

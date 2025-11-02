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
attack_class = 4


def nsl_kdd_one_class():
    pathTrain = os.path.join(fileDir, "./data/NSLKDD/NSLKDD_Train.csv")
    pathTest = os.path.join(fileDir, "./data/NSLKDD/NSLKDD_Test.csv")

    TrainData = np.genfromtxt(pathTrain, delimiter=",")
    TestData = np.genfromtxt(pathTest, delimiter=",")

    y_train = TrainData[:, -1]
    y_test = TestData[:, -1]

    y_train[y_train != attack_class] = 0
    y_test[y_test != attack_class] = 0
    y_train[y_train == attack_class] = 1
    y_test[y_test == attack_class] = 1

    X_train = TrainData[:, 0:-1]
    X_test = TestData[:, 0:-1]

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

    # print("Benign val: ", y_val[y_val == 0].shape)
    # print("Probe val: ", y_val[y_val == 1].shape)
    # print("DoS val: ", y_val[y_val == 2].shape)
    # print("R2L val: ", y_val[y_val == 3].shape)
    # print("R2L val: ", y_val[y_val == 4].shape)

    # print("Benign test: ", y_test[y_test == 0].shape)
    # print("Probe test: ", y_test[y_test == 1].shape)
    # print("DoS test: ", y_test[y_test == 2].shape)
    # print("R2L test: ", y_test[y_test == 3].shape)
    # print("R2L test: ", y_test[y_test == 4].shape)

    print("Splitted Size: ", (X_train.shape, y_train.shape), (X_val.shape, y_val.shape), (X_test.shape, y_test.shape))

    X_mal = X_train[y_train == 1]
    y_mal = y_train[y_train == 1]

    X_train = X_train[y_train == 0]
    y_train = y_train[y_train == 0]

    # X_val = X_val[y_val == 0]
    # y_val = y_val[y_val == 0]

    print("Final Size: ", (X_train.shape, y_train.shape), (X_val.shape, y_val.shape), (X_test.shape, y_test.shape))

    return X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal

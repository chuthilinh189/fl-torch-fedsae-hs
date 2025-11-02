import os
import pandas as pd
import numpy as np
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'
fileDir = os.path.dirname(os.path.abspath("__file__"))
data_dir = "./data/bot-iot"
test_size = 0.2
val_size = 0.125
random_state = 1
data_list = [
    "UNSW_2018_IoT_Botnet_Full5pc_1.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_2.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_3.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_4.csv",
]


def bot_iot():
    Data = pd.DataFrame()
    for i, item in enumerate(data_list):
        print("Loading data in file {}".format(item))
        item_data = pd.read_csv(
            os.path.join(data_dir, item),
            converters={"dport": partial(int, base=16), "sport": partial(int, base=16)},
        )
        Data = pd.concat([Data, item_data])

    #   Data = Data.sample(frac=0.1, random_state=random_state)

    y = Data["attack"]

    print(y[y == 1].shape, y[y == 0].shape)

    droppedColumn = [
        "pkSeqID",
        "stime",
        "flgs",
        "proto",
        "saddr",
        "daddr",
        "state",
        "ltime",
        "seq",
        "attack",
        "category",
        "subcategory",
    ]
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

    X_train = X_train[y_train == 0]
    y_train = y_train[y_train == 0]

    X_val = X_val[y_val == 0]
    y_val = y_val[y_val == 0]

    print("Final Size: ", (X_train.shape, y_train.shape), (X_val.shape, y_val.shape), (X_test.shape, y_test.shape))

    return X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal

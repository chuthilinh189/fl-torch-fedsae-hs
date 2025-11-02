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
features = [
    "srcip",
    "sport",
    "dstip",
    "dsport",
    "proto",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "service",
    "Sload",
    "Dload",
    "Spkts",
    "Dpkts",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "smeansz",
    "dmeansz",
    "trans_depth",
    "res_bdy_len",
    "Sjit",
    "Djit",
    "Stime",
    "Ltime",
    "Sintpkt",
    "Dintpkt",
    "tcprtt",
    "synack",
    "ackdat",
    "is_sm_ips_ports",
    "ct_state_ttl",
    "ct_flw_http_mthd",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "attack_cat",
    "Label",
]

cat_cols = [
    "proto",
    "state",
    "service",
    "attack_cat",
]


# 1: Final Size:  ((474345, 48), (474345,)) ((70000, 48), (70000,)) ((140001, 48), (140001,)) ((15655, 48), (15655,))


def unsw_big():
    Data = pd.DataFrame()
    for i in range(4):
        pathData = os.path.join(fileDir, "./data/unsw-big/UNSW-NB15_{}.csv".format(i + 1))
        sub_data = pd.read_csv(
            pathData,
            names=features,
            sep=",",
            header=None,
        )
        Data = pd.concat([Data, sub_data], ignore_index=True)

    print(Data.shape)
    Data = Data.drop("srcip", axis=1)
    Data = Data.drop("dstip", axis=1)
    Data.fillna(0)

    y = Data["Label"]
    y[y > 0] = 1

    X = Data.drop("Label", axis=1)
    X = pd.get_dummies(X, columns=cat_cols)
    print(X.sample())
    X.dropna()
    X.apply(lambda x: x if (isinstance(x, int) or isinstance(x, float)) else int(str(x), 16))

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

    print(
        "Final Size: ",
        (X_train.shape, y_train.shape),
        (X_val.shape, y_val.shape),
        (X_test.shape, y_test.shape),
        (X_mal.shape, y_mal.shape),
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal

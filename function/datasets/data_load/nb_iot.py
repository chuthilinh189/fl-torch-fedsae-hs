import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'
fileDir = os.path.dirname(os.path.abspath("__file__"))
data_dir = "./data/nb-iot"
test_size = 0.2
val_size = 0.125
random_state = 1
device_list = [
    "Ecobee_Thermostat",
    "Danmini_Doorbell",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]


def nb_iot():
    Data = pd.DataFrame()
    for i, device in enumerate(device_list):
        print("Loading data in device {}".format(device))
        # if i == 1:
        #    break
        # Get benign data
        benign_data = pd.read_csv(os.path.join(data_dir, device, "benign_traffic.csv"))
        benign_data["class"] = 0

        Data = pd.concat([Data, benign_data])

        # Get malicious data
        g_attack_data_list = [
            os.path.join(data_dir, device, "gafgyt_attacks", f)
            for f in os.listdir(os.path.join(data_dir, device, "gafgyt_attacks"))
        ]
        if device == "Ennio_Doorbell" or device == "Samsung_SNH_1011_N_Webcam":
            attack_data_list = g_attack_data_list
        else:
            m_attack_data_list = [
                os.path.join(data_dir, device, "mirai_attacks", f)
                for f in os.listdir(os.path.join(data_dir, device, "mirai_attacks"))
            ]
            attack_data_list = g_attack_data_list + m_attack_data_list

        attack_data = pd.concat([pd.read_csv(f) for f in attack_data_list], ignore_index=True)
        attack_data["class"] = 1
        attack_data = attack_data.sample(frac=0.5, random_state=random_state)
        Data = pd.concat([Data, attack_data])

    Data = Data.sample(frac=0.05, random_state=random_state)

    y = Data["class"]
    X = Data.drop("class", axis=1)
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

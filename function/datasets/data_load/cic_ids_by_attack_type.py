import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

fileDir = os.path.dirname(os.path.abspath("__file__"))

test_size = 0.2
val_size = 0.125
random_state = 1


def cic_ids_by_attack_type():
    pathTrain = os.path.join(fileDir, "./data/cic-ids/CIC-IDS-2017-Train-10percent.csv")
    pathTest = os.path.join(fileDir, "./data/cic-ids/CIC-IDS-2017-Test-10percent.csv")

    TrainData = pd.read_csv(pathTrain, sep=",", header=None)
    TestData = pd.read_csv(pathTest, sep=",", header=None)

    # Loại bỏ giá trị NaN / inf
    TrainData.replace([np.inf, -np.inf], np.nan, inplace=True)
    TrainData.dropna(inplace=True)
    TestData.replace([np.inf, -np.inf], np.nan, inplace=True)
    TestData.dropna(inplace=True)

    Data = pd.concat([TrainData, TestData])
    y = Data[85]

    droppedColumn = [0, 1, 2, 4, 7, 85]
    X = Data
    for idx in droppedColumn:
        X = X.drop([idx], axis=1)

    # In số lượng mẫu từng loại attack type
    print("📊 Tổng số mẫu theo attack type:")
    for name, count in y.value_counts().items():
        print(f"  {name:20}: {count} mẫu")

    print("Original Size: ", (X.shape, y.shape))

    # Chia train/test
    X_train, X_test, y_train_str, y_test_str = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Ánh xạ label cho train/val: chỉ giữ BENIGN, DDoS, DoS Hulk
    valid_attacks = ["DDoS", "DoS Hulk"]
    def map_train_label(label):
        if label == "BENIGN":
            return 0
        elif label in valid_attacks:
            return 1
        else:
            return -1  # loại bỏ

    y_train = y_train_str.map(map_train_label)
    train_mask = y_train != -1
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    # === Ánh xạ attack type trong y_test sang mã số (BENIGN = 0, DDoS = 1, ...)
    unique_labels = sorted(y_test_str.unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    print("📘 Attack type → ID:")
    for k, v in label2id.items():
        print(f"  {k:20}: {v}")
    y_test = y_test_str.map(label2id)

    # Convert về array
    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.int32)

    # Chuẩn hóa dữ liệu
    mmsc = MinMaxScaler()
    X_train = mmsc.fit_transform(X_train)
    X_test = mmsc.transform(X_test)

    # Chia val từ train
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    # Tạo tập dữ liệu tấn công để sử dụng (train)
    X_mal = X_train[y_train == 1]
    y_mal = y_train[y_train == 1]

    print("Splitted Size: ", (X_train.shape, y_train.shape), (X_val.shape, y_val.shape), (X_test.shape, y_test.shape))
    print("Final Size: ", (X_mal.shape, y_mal.shape))

    return X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal

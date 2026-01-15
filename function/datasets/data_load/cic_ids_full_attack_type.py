import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

fileDir = os.path.dirname(os.path.abspath("__file__"))

test_size = 0.2
val_size = 0.125
random_state = 1


def _build_label_mapping(labels: pd.Series):
    """Map string labels to deterministic integers (BENIGN -> 0, others sorted)."""
    unique_labels = sorted(set(labels.unique()))
    mapping = {}
    # Đảm bảo BENIGN là 0 nếu tồn tại
    if "BENIGN" in unique_labels:
        mapping["BENIGN"] = 0
        unique_labels.remove("BENIGN")
    # Các attack được gán từ 1 trở đi theo thứ tự alphabet còn lại
    for idx, lab in enumerate(unique_labels, start=1 if "BENIGN" in mapping else 0):
        mapping[lab] = idx
    return mapping


def cic_ids_full_attack_type():
    pathTrain = os.path.join(fileDir, "./data/cic-ids/CIC-IDS-2017-Train-10percent.csv")
    pathTest = os.path.join(fileDir, "./data/cic-ids/CIC-IDS-2017-Test-10percent.csv")

    TrainData = pd.read_csv(pathTrain, sep=",", header=None)
    TestData = pd.read_csv(pathTest, sep=",", header=None)

    # Clean inf/nan
    for df in (TrainData, TestData):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

    Data = pd.concat([TrainData, TestData], ignore_index=True)

    # Lọc bỏ hai lớp hiếm theo tên chuỗi rồi giữ nguyên nhãn tấn công
    rare_labels = ["Infiltration", "Heartbleed"]
    mask = ~Data[85].isin(rare_labels)
    Data_filtered = Data.loc[mask].reset_index(drop=True)

    y = Data_filtered[85]
    label_map = _build_label_mapping(y)
    y = y.map(label_map).astype(np.int64)

    # Bỏ các cột không dùng như bản nhị phân
    droppedColumn = [0, 1, 2, 4, 7, 85]
    X = Data_filtered.drop(columns=droppedColumn)

    print("Original Size: ", (X.shape, y.shape))
    print(f"Label mapping (string->int): {label_map}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    mmsc = MinMaxScaler()
    X_train = mmsc.fit_transform(X_train)
    X_test = mmsc.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )  # 0.125 x 0.8 = 0.1

    print("Splitted Size: ", (X_train.shape, y_train.shape), (X_val.shape, y_val.shape), (X_test.shape, y_test.shape))

    # Mal pool: mọi nhãn khác 0
    X_mal = X_train[y_train != 0]
    y_mal = y_train[y_train != 0]

    print(
        "Final Size: ",
        (X_train.shape, y_train.shape),
        (X_val.shape, y_val.shape),
        (X_test.shape, y_test.shape),
        (X_mal.shape, y_mal.shape),
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal

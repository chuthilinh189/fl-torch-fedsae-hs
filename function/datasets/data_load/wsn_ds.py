import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None
fileDir = os.path.dirname(os.path.abspath("__file__"))

test_size = 0.2
val_size = 0.125
random_state = 1

def wsn_ds():
    path = os.path.join(fileDir, "./data/wsn-ds/WSN-DS.csv")  # chỉnh lại path nếu cần
    df = pd.read_csv(path)

    # Nhãn nhị phân: normal = 0, còn lại = 1
    df['label'] = df['Attack type'].apply(lambda x: 0 if str(x).lower() == "normal" else 1)

    # Loại bỏ cột không cần thiết
    df = df.drop(columns=[' id', 'Attack type'])

    # Tách đặc trưng và nhãn
    X = df.drop('label', axis=1)
    y = df['label']

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)
    
    # Chuẩn hóa
    mmsc = MinMaxScaler()
    X_train = mmsc.fit_transform(X_train)
    X_test = mmsc.transform(X_test)

    # Nếu cần, lọc dữ liệu train, val chỉ có lớp 0
    # X_train = X_train[y_train == 0]
    # y_train = y_train[y_train == 0]

    # Chia val từ train
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    # Dữ liệu bất thường từ train (nếu dùng trong FL)
    X_mal = X_train[y_train == 1]
    y_mal = y_train[y_train == 1]

    return X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal

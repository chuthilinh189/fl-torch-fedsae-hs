import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None
fileDir = os.path.dirname(os.path.abspath("__file__"))

test_size = 0.2
val_size = 0.125
random_state = 1

def ton_iot_network():
    # Đường dẫn đến file CSV (bạn chỉnh nếu đặt ở nơi khác)
    path = os.path.join(fileDir, "./data/ton-iot/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv")
    df = pd.read_csv(path)

    # Gán nhãn nhị phân: normal = 0, attack = 1
    df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)

    # Loại bỏ các cột chuỗi hoặc không cần thiết (dựa theo ảnh bạn gửi)
    drop_cols = [
        'type', 'src_ip', 'dst_ip', 'proto', 'service', 'conn_state',
        'dns_query', 'dns_AA', 
        'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_version', 'ssl_cipher',
        'ssl_resumed', 'ssl_established', 'ssl_subject', 'ssl_issuer',
        'http_trans_depth', 'http_method', 'http_uri', 'http_version', 'http_user_agent',
        'http_orig_mime_types', 'http_resp_mime_types',
        'weird_name', 'weird_addl', 'weird_notice'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

    # Chỉ giữ lại các cột số (numeric)
    df = df.select_dtypes(include=[np.number])

    # Tách đặc trưng và nhãn
    X = df.drop(columns=['label'])
    y = df['label']

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)

    # Chuẩn hóa Min-Max
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Nếu cần, lọc dữ liệu train, val chỉ có lớp 0
    # X_train = X_train[y_train == 0]
    # y_train = y_train[y_train == 0]
    
    # Chia tập val từ train
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

    # Dữ liệu bất thường để dùng trong FL
    X_mal = X_train[y_train == 1]
    y_mal = y_train[y_train == 1]

    print("📊 Final data shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)
    print("Malicious (for FL poison):", X_mal.shape, y_mal.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal

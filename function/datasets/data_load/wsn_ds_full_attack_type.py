import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

fileDir = os.path.dirname(os.path.abspath("__file__"))

test_size = 0.2
val_size = 0.125
random_state = 1


def wsn_ds_full_attack_type():
    """Load WSN-DS dataset keeping multi-class labels intact.

    Assumes CSV with features in all columns except last (label). Labels are ints.
    Returns: X_train, y_train, X_val, y_val, X_test, y_test, X_mal, y_mal
    """
    path = os.path.join(fileDir, "./data/wsn-ds/WSN-DS.csv")
    df = pd.read_csv(path)

    # Clean column names (strip spaces/newlines) for consistent processing
    df.columns = [col.strip().replace("\n", "") for col in df.columns]

    # Preserve multi-class attack labels as integer codes
    y, label_names = pd.factorize(df["Attack type"])
    normal_idx = np.where(label_names == "Normal")[0]
    normal_code = int(normal_idx[0]) if len(normal_idx) > 0 else -1

    # Drop non-feature columns
    df = df.drop(columns=["id", "Attack type"])
    X = df.astype(np.float64).values
    y = np.asarray(y, dtype=np.int64)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Val split from train
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )  # 0.125 * 0.8 = 0.1 of total

    # Malicious pool: label != 0
    # Malicious pool: labels different from Normal (if present)
    if normal_code >= 0:
        mal_mask = y_train != normal_code
    else:
        mal_mask = np.ones_like(y_train, dtype=bool)

    X_mal = X_train[mal_mask]
    y_mal = y_train[mal_mask]

    return (
        np.asarray(X_train, dtype=np.float64),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(X_val, dtype=np.float64),
        np.asarray(y_val, dtype=np.int64),
        np.asarray(X_test, dtype=np.float64),
        np.asarray(y_test, dtype=np.int64),
        np.asarray(X_mal, dtype=np.float64),
        np.asarray(y_mal, dtype=np.int64),
    )

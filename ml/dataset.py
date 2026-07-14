import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys
import logging

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.features import FEATURE_COLS, LABEL_COL


class StockDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_datasets(df: pd.DataFrame, val_ratio=0.2, scaler_path="models/scaler.pkl",
                   fit_scaler=True):
    df = df.copy()
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int64)
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid_mask], y[valid_mask]

    split_idx = int(len(X) * (1 - val_ratio))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    os.makedirs(os.path.dirname(scaler_path) or "models", exist_ok=True)
    if fit_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
    else:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)}")
    return StockDataset(X_train, y_train), StockDataset(X_val, y_val), scaler


def build_datasets_from_frames(train_df: pd.DataFrame, test_df: pd.DataFrame,
                               scaler_path="models/scaler.pkl", save_scaler=True):
    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df[LABEL_COL].values.astype(np.int64)
    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df[LABEL_COL].values.astype(np.int64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path) or "models", exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

    return StockDataset(X_train, y_train), StockDataset(X_test, y_test), scaler


def prepare_inference_input(df: pd.DataFrame, scaler: StandardScaler) -> torch.Tensor:
    last_row = df[FEATURE_COLS].iloc[[-1]].values.astype(np.float32)
    return torch.tensor(scaler.transform(last_row), dtype=torch.float32)


def create_dataloaders(train_ds, val_ds, batch_size=64):
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )

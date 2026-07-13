import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

PREDICTION_HORIZON = 5
BULL_THRESHOLD = 0.02
BEAR_THRESHOLD = -0.02


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).rename("rsi")


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": macd_line - signal_line})


def compute_sma(series: pd.Series, windows=[20, 50]) -> pd.DataFrame:
    return pd.DataFrame({f"sma_{w}": series.rolling(window=w).mean() for w in windows})


def compute_volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    roll_mean = volume.rolling(window=window).mean()
    roll_std = volume.rolling(window=window).std()
    return ((volume - roll_mean) / roll_std.replace(0, np.nan)).rename("volume_zscore")


def compute_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    log_returns = np.log(close / close.shift(1))
    return (log_returns.rolling(window=window).std() * np.sqrt(252)).rename("volatility")


def compute_label(close: pd.Series, horizon=PREDICTION_HORIZON,
                  bull_threshold=BULL_THRESHOLD, bear_threshold=BEAR_THRESHOLD) -> pd.Series:
    future_return = close.shift(-horizon) / close - 1
    label = pd.Series(np.nan, index=close.index, name="label")
    label[future_return > bull_threshold] = 2
    label[future_return < bear_threshold] = 0
    label[(future_return >= bear_threshold) & (future_return <= bull_threshold)] = 1
    return label


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["rsi"] = compute_rsi(out["close"])
    out = pd.concat([out, compute_macd(out["close"])], axis=1)
    out = pd.concat([out, compute_sma(out["close"])], axis=1)
    out["price_vs_sma20"] = (out["close"] - out["sma_20"]) / out["sma_20"]
    out["price_vs_sma50"] = (out["close"] - out["sma_50"]) / out["sma_50"]
    out["volume_zscore"] = compute_volume_zscore(out["volume"])
    out["volatility"] = compute_volatility(out["close"])
    out["daily_return"] = np.log(out["close"] / out["close"].shift(1))
    out["label"] = compute_label(out["close"])
    out = out.dropna().reset_index(drop=True)
    logger.info(f"Features : {len(out)} lignes")
    return out


FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "price_vs_sma20", "price_vs_sma50",
    "volume_zscore", "volatility", "daily_return",
]
LABEL_COL = "label"
NUM_CLASSES = 3

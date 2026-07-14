import os
import numpy as np
import pandas as pd
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PREDICTION_HORIZON = int(os.getenv("PREDICTION_HORIZON", 5))
# Seuils par défaut calibrés pour un horizon de 5 jours ; mis à l'échelle en sqrt(horizon)
# (hypothèse de marche aléatoire) si un horizon différent est configuré, pour garder un
# équilibre de classes comparable plutôt que d'écraser la classe "Neutre".
_HORIZON_SCALE = (PREDICTION_HORIZON / 5) ** 0.5
BULL_THRESHOLD = float(os.getenv("BULL_THRESHOLD", 0.02 * _HORIZON_SCALE))
BEAR_THRESHOLD = float(os.getenv("BEAR_THRESHOLD", -0.02 * _HORIZON_SCALE))


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


def compute_bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({
        "bb_middle": middle,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": (upper - lower) / middle.replace(0, np.nan),
        "bb_position": (series - lower) / (upper - lower).replace(0, np.nan),
    })


def compute_label(close: pd.Series, horizon=PREDICTION_HORIZON,
                  bull_threshold=BULL_THRESHOLD, bear_threshold=BEAR_THRESHOLD) -> pd.Series:
    future_return = close.shift(-horizon) / close - 1
    label = pd.Series(np.nan, index=close.index, name="label")
    label[future_return > bull_threshold] = 2
    label[future_return < bear_threshold] = 0
    label[(future_return >= bear_threshold) & (future_return <= bull_threshold)] = 1
    return label


def build_features(df: pd.DataFrame, news_sentiment: pd.DataFrame = None) -> pd.DataFrame:
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["rsi"] = compute_rsi(out["close"])
    out = pd.concat([out, compute_macd(out["close"])], axis=1)
    out = pd.concat([out, compute_sma(out["close"])], axis=1)
    out = pd.concat([out, compute_bollinger(out["close"])], axis=1)
    out["price_vs_sma20"] = (out["close"] - out["sma_20"]) / out["sma_20"]
    out["price_vs_sma50"] = (out["close"] - out["sma_50"]) / out["sma_50"]
    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()
    out["ema_ratio_12_26"] = (out["ema_12"] - out["ema_26"]) / out["ema_26"]
    out["volume_zscore"] = compute_volume_zscore(out["volume"])
    out["volatility"] = compute_volatility(out["close"])
    out["daily_return"] = np.log(out["close"] / out["close"].shift(1))
    out["return_3"] = out["close"].pct_change(3)
    out["return_5"] = out["close"].pct_change(5)
    out["return_10"] = out["close"].pct_change(10)
    out["momentum_5"] = out["close"] / out["close"].shift(5) - 1
    out["momentum_10"] = out["close"] / out["close"].shift(10) - 1
    out["volume_change_1"] = out["volume"].pct_change(1)
    out["volume_change_5"] = out["volume"].pct_change(5)
    out["spread"] = (out["high"] - out["low"]) / out["close"]
    out["close_open_gap"] = (out["close"] - out["open"]) / out["open"]

    if news_sentiment is not None and not news_sentiment.empty:
        out = out.merge(news_sentiment, on="date", how="left")
        # Neutre (0) les jours sans actualité datée disponible dans l'historique.
        out["news_sentiment"] = out["news_sentiment"].fillna(0.0)
    else:
        out["news_sentiment"] = 0.0

    out["label"] = compute_label(out["close"])
    out = out.dropna(subset=[c for c in out.columns if c != "news_sentiment"]).reset_index(drop=True)
    logger.info(f"Features : {len(out)} lignes")
    return out


FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "price_vs_sma20", "price_vs_sma50",
    "ema_ratio_12_26", "bb_position", "bb_width",
    "volume_zscore", "volatility", "daily_return",
    "return_3", "return_5", "return_10",
    "momentum_5", "momentum_10", "volume_change_1", "volume_change_5",
    "spread", "close_open_gap",
]
# "news_sentiment" est calculé par build_features() mais volontairement exclu de
# l'entraînement : la couverture historique gratuite d'Alpha Vantage est incomplète
# (pas de données avant 2022, et plafond de 1000 articles/appel qui coupe l'historique
# récent des tickers à fort volume de news comme NVDA/INTC/AMD) — l'inclure introduisait
# plus de bruit que de signal réel (voir walk-forward : edge moyen dégradé de -16 à -25 pts).
LABEL_COL = "label"
NUM_CLASSES = 3

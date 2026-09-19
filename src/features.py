"""計算不使用未來資料的技術指標特徵。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(
    ohlcv: pd.DataFrame,
    ma_windows: tuple[int, ...] = (5, 20),
    ema_windows: tuple[int, ...] = (5, 20),
    rsi_window: int = 14,
    atr_window: int = 14,
    volatility_window: int = 20,
    volume_window: int = 20,
) -> pd.DataFrame:
    """回傳原始資料加上 MA、EMA、RSI、MACD、ATR 等欄位。"""
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(ohlcv.columns):
        raise ValueError(f"缺少欄位: {sorted(required.difference(ohlcv.columns))}")
    frame = ohlcv.copy().sort_index()
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    for window in ma_windows:
        frame[f"ma_{window}"] = close.rolling(window).mean()
    for window in ema_windows:
        frame[f"ema_{window}"] = close.ewm(span=window, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_window).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    rs = gain / loss.replace(0, np.nan)
    frame["rsi"] = 100 - (100 / (1 + rs))
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    frame["macd"] = ema_fast - ema_slow
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    frame["atr"] = true_range.rolling(atr_window).mean()
    frame["volatility"] = close.pct_change().rolling(volatility_window).std()
    frame["volume_ma"] = frame["volume"].rolling(volume_window).mean()
    frame["volume_ratio"] = frame["volume"] / frame["volume_ma"]
    frame["return_1"] = close.pct_change()
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """取得可供模型使用的數值特徵欄位。"""
    excluded = {"open", "high", "low", "close", "volume", "label", "future_return"}
    return [column for column in frame.columns if column not in excluded]

"""建立未來 N 根 K 棒方向標籤。"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_TO_ID = {"SELL": 0, "HOLD": 1, "BUY": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def add_direction_labels(
    frame: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.001,
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
) -> pd.DataFrame:
    """依未來報酬建立 BUY、HOLD、SELL 三分類標籤。"""
    if horizon < 1 or threshold < 0:
        raise ValueError("horizon 必須為正數，threshold 不可為負數")
    buy_threshold = threshold if buy_threshold is None else buy_threshold
    sell_threshold = threshold if sell_threshold is None else sell_threshold
    if buy_threshold < 0 or sell_threshold < 0:
        raise ValueError("buy_threshold 與 sell_threshold 不可為負數")
    result = frame.copy()
    result["future_return"] = result["close"].shift(-horizon) / result["close"] - 1
    result["label"] = np.select(
        [result["future_return"] > buy_threshold, result["future_return"] < -sell_threshold],
        ["BUY", "SELL"],
        default="HOLD",
    )
    return result.dropna(subset=["future_return"]).copy()


def encode_labels(labels: pd.Series) -> pd.Series:
    """將文字標籤轉換為模型使用的整數。"""
    unknown = set(labels.dropna().unique()).difference(LABEL_TO_ID)
    if unknown:
        raise ValueError(f"未知標籤: {sorted(unknown)}")
    return labels.map(LABEL_TO_ID).astype(int)

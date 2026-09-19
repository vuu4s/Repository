"""提供台股分鐘 K 資料載入與可重現的 mock 資料。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_mock_ohlcv(rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """產生不連接外部服務的分鐘 OHLCV 資料。"""
    if rows < 10:
        raise ValueError("rows 必須至少為 10")
    rng = np.random.default_rng(seed)
    minutes_per_session = 271
    sessions = pd.date_range(
        "2024-01-02", periods=(rows + minutes_per_session - 1) // minutes_per_session, freq="B"
    )
    session_indexes = [
        pd.date_range(f"{session.date()} 09:00", f"{session.date()} 13:30", freq="min")
        for session in sessions
    ]
    index = pd.DatetimeIndex(np.concatenate(session_indexes))[:rows]
    returns = rng.normal(0.00003, 0.0015, rows)
    close = 100 * np.exp(np.cumsum(returns))
    open_price = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0003, rows))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0, 0.0015, rows))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0, 0.0015, rows))
    volume = rng.lognormal(mean=10, sigma=0.35, size=rows).astype(int)
    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """載入包含 datetime、OHLCV 欄位的 CSV 分鐘資料。"""
    frame = pd.read_csv(path, parse_dates=["datetime"])
    frame = frame.set_index("datetime").sort_index()
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV 缺少欄位: {sorted(missing)}")
    return frame[sorted(required)]

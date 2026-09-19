import pandas as pd

from src.data_loader import generate_mock_ohlcv
from src.features import add_features


def test_features_are_created_without_missing_values():
    frame = add_features(generate_mock_ohlcv(100))
    assert {"ma_5", "ema_20", "rsi", "macd", "atr", "volatility", "volume_ratio"}.issubset(frame.columns)
    assert not frame.isna().any().any()
    assert isinstance(frame.index, pd.DatetimeIndex)

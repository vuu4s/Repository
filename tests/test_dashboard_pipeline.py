import numpy as np
import pandas as pd

from src.dashboard_pipeline import predictions_to_signals, run_prediction_backtest


def test_dashboard_mapping_and_long_only_backtest():
    index = pd.date_range("2024-01-01", periods=4, freq="min")
    prices = pd.Series([100.0, 101.0, 100.0, 102.0], index=index)
    signals = predictions_to_signals(np.array([2, 2, 1, 0]), index)
    assert signals.tolist() == ["BUY", "BUY", "HOLD", "SELL"]
    result, summary = run_prediction_backtest(
        prices,
        np.array([2, 2, 1, 0]),
        index,
        initial_cash=1000,
        fee_rate=0.001425,
        tax_rate=0.003,
        slippage_rate=0.0005,
        position_size=1,
    )
    assert set(result["position"].unique()).issubset({0, 1})
    assert summary["total_trades"] == 1
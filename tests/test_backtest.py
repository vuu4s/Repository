import pandas as pd

from src.backtest import backtest
from src.metrics import strategy_metrics


def test_backtest_applies_costs_and_returns_metrics():
    index = pd.date_range("2024-01-01", periods=4, freq="min")
    prices = pd.Series([100.0, 101.0, 100.0, 102.0], index=index)
    signals = pd.Series(["BUY", "BUY", "HOLD", "SELL"], index=index)
    equity = backtest(prices, signals, initial_cash=1000, position_size=1)
    result = strategy_metrics(equity["equity"], periods_per_year=4)
    assert len(equity) == 4
    assert result["strategy_return"] < 0
    assert "maximum_drawdown" in result

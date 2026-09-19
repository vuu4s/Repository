"""簡化的當沖歷史回測引擎。"""

from __future__ import annotations

import pandas as pd


def backtest(
    prices: pd.Series,
    signals: pd.Series,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001425,
    tax_rate: float = 0.003,
    slippage_rate: float = 0.0005,
    position_size: float = 1.0,
) -> pd.DataFrame:
    """以訊號持有多單部位，並在訊號變更時結算交易成本。"""
    if not prices.index.equals(signals.index):
        raise ValueError("prices 與 signals 的 index 必須相同")
    cash = float(initial_cash)
    position = 0
    rows: list[dict[str, float]] = []
    for timestamp, price in prices.items():
        signal = str(signals.loc[timestamp]).upper()
        target = 1 if signal == "BUY" else 0
        if target != position:
            turnover = abs(target - position) * float(price) * position_size
            sell_tax = tax_rate if position == 1 and target == 0 else 0
            cost = turnover * (fee_rate + sell_tax + slippage_rate)
            cash -= cost
            position = target
        equity = cash + position * float(price) * position_size
        rows.append({"datetime": timestamp, "price": float(price), "position": position, "cash": cash, "equity": equity})
    return pd.DataFrame(rows).set_index("datetime")

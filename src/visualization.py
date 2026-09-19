"""研究用價格、訊號、權益與回撤圖表。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd


def plot_ohlcv(frame: pd.DataFrame, title: str = "價格與成交量") -> plt.Figure:
    """繪製 OHLC K 線與成交量。"""
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=(3, 1))
    date_numbers = mdates.date2num(frame.index.to_pydatetime())
    candle_width = 0.0005
    for date_number, (_, row) in zip(date_numbers, frame.iterrows()):
        color = "#27ae60" if row["close"] >= row["open"] else "#c0392b"
        axes[0].vlines(date_number, row["low"], row["high"], color=color, linewidth=0.7)
        body_bottom = min(row["open"], row["close"])
        body_height = max(abs(row["close"] - row["open"]), 1e-8)
        axes[0].add_patch(
            Rectangle(
                (date_number - candle_width / 2, body_bottom),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.85,
            )
        )
    axes[0].set_title(title)
    axes[0].set_ylabel("價格")
    axes[0].grid(alpha=0.2)
    axes[1].bar(date_numbers, frame["volume"], width=candle_width, color="#7f8c8d")
    axes[1].set_ylabel("Volume")
    axes[1].xaxis_date()
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_indicators(frame: pd.DataFrame) -> plt.Figure:
    """繪製 MA、EMA、RSI、MACD、ATR 與波動率。"""
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(frame.index, frame["close"], label="Close")
    for column in ["ma_5", "ma_20", "ema_5", "ema_20"]:
        if column in frame:
            axes[0].plot(frame.index, frame[column], label=column)
    axes[0].legend(loc="upper left", ncol=5)
    axes[0].set_title("技術指標")
    axes[1].plot(frame.index, frame["rsi"], label="RSI", color="#e67e22")
    axes[1].axhline(70, color="#c0392b", linestyle="--", alpha=0.5)
    axes[1].axhline(30, color="#27ae60", linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper left")
    axes[2].plot(frame.index, frame["macd"], label="MACD")
    axes[2].plot(frame.index, frame["macd_signal"], label="Signal")
    axes[2].legend(loc="upper left")
    axes[3].plot(frame.index, frame["atr"], label="ATR")
    axes[3].plot(frame.index, frame["volatility"], label="Volatility")
    axes[3].legend(loc="upper left")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_signals(result: pd.DataFrame, title: str = "回測成交訊號") -> plt.Figure:
    """繪製價格與實際部位變更成交點。"""
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(result.index, result["price"], label="Price", color="#34495e")
    bought = result["position"].diff().eq(1)
    sold = result["position"].diff().eq(-1)
    axis.scatter(result.index[bought], result.loc[bought, "price"], marker="^", color="#27ae60", label="BUY")
    axis.scatter(result.index[sold], result.loc[sold, "price"], marker="v", color="#c0392b", label="SELL")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_equity(result: pd.DataFrame) -> plt.Figure:
    """繪製資金權益曲線與 drawdown。"""
    equity = result["equity"]
    drawdown = equity / equity.cummax() - 1
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(result.index, equity, color="#2980b9")
    axes[0].set_title("資金權益曲線")
    axes[0].grid(alpha=0.2)
    axes[1].fill_between(result.index, drawdown, 0, color="#c0392b", alpha=0.35)
    axes[1].set_title("Drawdown")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    return figure
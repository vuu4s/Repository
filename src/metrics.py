"""分類與策略績效指標。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true: pd.Series, y_pred: pd.Series, probabilities: np.ndarray | None = None) -> dict[str, float]:
    """計算 Accuracy、Precision、Recall、F1 與可用時的 AUC。"""
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc": float("nan"),
    }
    if probabilities is not None:
        try:
            result["auc"] = float(roc_auc_score(y_true, probabilities, multi_class="ovr"))
        except ValueError:
            pass
    return result


def strategy_metrics(equity: pd.Series, periods_per_year: int = 252 * 390) -> dict[str, float]:
    """計算策略報酬、勝率、Profit Factor、最大回撤與 Sharpe。"""
    returns = equity.pct_change().fillna(0)
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() else 0.0
    return {
        "strategy_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "win_rate": float((returns > 0).mean()),
        "profit_factor": float(gains / losses) if losses else float("inf"),
        "maximum_drawdown": float(drawdown.min()),
        "sharpe_ratio": float(sharpe),
    }

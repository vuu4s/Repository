"""Streamlit 控制台使用的研究流程編排。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .backtest import backtest
from .features import add_features, feature_columns
from .labels import ID_TO_LABEL
from .lstm import LSTMClassifier, make_sequences, train_lstm_history
from .metrics import classification_metrics, strategy_metrics
from .ml_baseline import TemporalSplit, fit_baselines, temporal_split, transform_split


@dataclass
class AnalysisBundle:
    """已完成特徵、標籤與時間切分的分析資料。"""

    raw: pd.DataFrame
    frame: pd.DataFrame
    split: TemporalSplit
    feature_names: list[str]
    scaler: object | None = None


def prepare_dataset(
    raw: pd.DataFrame,
    feature_config: dict[str, object],
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
    train_ratio: float,
    validation_ratio: float,
) -> AnalysisBundle:
    """建立不使用未來特徵的資料集並依時間切分。"""
    from .labels import add_direction_labels

    frame = add_features(raw, **feature_config)
    frame = add_direction_labels(
        frame,
        horizon=horizon,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    split = temporal_split(frame, train_ratio=train_ratio, validation_ratio=validation_ratio)
    return AnalysisBundle(raw=raw, frame=frame, split=split, feature_names=feature_columns(frame))


def fit_baseline_analysis(bundle: AnalysisBundle, n_estimators: int = 100, random_state: int = 42) -> dict[str, object]:
    """只以 training fit scaler 與 baseline，回傳 test metrics。"""
    models, scaler = fit_baselines(
        bundle.split,
        bundle.feature_names,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    bundle.scaler = scaler
    test_features, test_labels = transform_split(bundle.split.test, bundle.feature_names, scaler)
    evaluations: dict[str, object] = {}
    for name, model in models.items():
        predictions = model.predict(test_features.to_numpy())
        probabilities = model.predict_proba(test_features.to_numpy())
        evaluations[name] = {
            "model": model,
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": classification_metrics(test_labels, predictions, probabilities),
        }
    return {"models": models, "scaler": scaler, "evaluations": evaluations}


def fit_lstm_analysis(
    bundle: AnalysisBundle,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    random_seed: int = 42,
) -> dict[str, object]:
    """以 training fit scaler，訓練 CPU LSTM 並評估 validation/test。"""
    if bundle.scaler is None:
        fit_baseline_analysis(bundle, random_state=random_seed)
    assert bundle.scaler is not None
    train_features, train_labels = transform_split(bundle.split.train, bundle.feature_names, bundle.scaler)
    validation_features, validation_labels = transform_split(bundle.split.validation, bundle.feature_names, bundle.scaler)
    test_features, test_labels = transform_split(bundle.split.test, bundle.feature_names, bundle.scaler)
    train_x, train_y = make_sequences(train_features.to_numpy(dtype=np.float32), train_labels.to_numpy(), sequence_length)
    validation_x, validation_y = make_sequences(
        validation_features.to_numpy(dtype=np.float32), validation_labels.to_numpy(), sequence_length
    )
    test_x, test_y = make_sequences(test_features.to_numpy(dtype=np.float32), test_labels.to_numpy(), sequence_length)
    torch.manual_seed(random_seed)
    model = LSTMClassifier(input_size=len(bundle.feature_names), hidden_size=hidden_size, num_layers=num_layers)
    model, history = train_lstm_history(
        model,
        train_x,
        train_y,
        validation_x,
        validation_y,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(torch.as_tensor(test_x, dtype=torch.float32))
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    return {
        "model": model,
        "history": history,
        "predictions": predictions,
        "probabilities": probabilities,
        "labels": test_y,
        "metrics": classification_metrics(test_y, predictions, probabilities),
        "sequence_length": sequence_length,
        "test_index": bundle.split.test.index[sequence_length:],
    }


def predictions_to_signals(predictions: np.ndarray | list[int], index: pd.Index) -> pd.Series:
    """將模型整數預測統一轉成 SELL/HOLD/BUY。"""
    if len(predictions) != len(index):
        raise ValueError("predictions 與 index 長度必須相同")
    return pd.Series([ID_TO_LABEL[int(value)] for value in predictions], index=index)


def run_prediction_backtest(
    prices: pd.Series,
    predictions: np.ndarray | list[int],
    index: pd.Index,
    initial_cash: float,
    fee_rate: float,
    tax_rate: float,
    slippage_rate: float,
    position_size: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """將預測轉成訊號並執行只含多單的歷史回測。"""
    signals = predictions_to_signals(predictions, index)
    result = backtest(
        prices.loc[index],
        signals,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        tax_rate=tax_rate,
        slippage_rate=slippage_rate,
        position_size=position_size,
    )
    return result, trade_summary(result, fee_rate, tax_rate, slippage_rate)


def trade_summary(result: pd.DataFrame, fee_rate: float, tax_rate: float, slippage_rate: float) -> dict[str, float]:
    """計算完整 round-trip 交易統計與策略統計。"""
    entry_price: float | None = None
    trade_returns: list[float] = []
    position_changes = result["position"].diff().fillna(result["position"])
    for timestamp in result.index:
        position_change = position_changes.loc[timestamp]
        price = float(result.loc[timestamp, "price"])
        if position_change == 1:
            entry_price = price
        elif position_change == -1 and entry_price is not None:
            buy_cost = entry_price * (1 + fee_rate + slippage_rate)
            sell_value = price * (1 - fee_rate - tax_rate - slippage_rate)
            trade_returns.append(sell_value / buy_cost - 1)
            entry_price = None
    strategy = strategy_metrics(result["equity"])
    wins = [value for value in trade_returns if value > 0]
    losses = [-value for value in trade_returns if value < 0]
    return {
        **strategy,
        "total_trades": float(len(trade_returns)),
        "win_rate": float(len(wins) / len(trade_returns)) if trade_returns else 0.0,
        "average_trade_return": float(np.mean(trade_returns)) if trade_returns else 0.0,
        "profit_factor": float(sum(wins) / sum(losses)) if losses else float("inf"),
    }
"""只使用當時已知資料的 expanding-window walk-forward 評估。"""

from __future__ import annotations

import pandas as pd

from .backtest import backtest
from .labels import encode_labels
from .metrics import classification_metrics, strategy_metrics
from .ml_baseline import TemporalSplit, fit_baselines, transform_split


def walk_forward_evaluate(
    frame: pd.DataFrame,
    feature_names: list[str],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    n_estimators: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """以 expanding train window 逐步向前評估 baseline 模型。"""
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("walk-forward 比例必須為正數")
    train_size = int(len(frame) * train_ratio)
    validation_size = int(len(frame) * validation_ratio)
    test_size = int(len(frame) * test_ratio)
    rows: list[dict[str, float | int | str]] = []
    fold = 0
    train_end = train_size
    while train_end + validation_size + test_size <= len(frame):
        validation_end = train_end + validation_size
        test_end = validation_end + test_size
        split = TemporalSplit(frame.iloc[:train_end], frame.iloc[train_end:validation_end], frame.iloc[validation_end:test_end])
        models, scaler = fit_baselines(split, feature_names, random_state=random_state, n_estimators=n_estimators)
        test_features, test_labels = transform_split(split.test, feature_names, scaler)
        for model_name, model in models.items():
            predictions = model.predict(test_features.to_numpy())
            probabilities = model.predict_proba(test_features.to_numpy())
            classification = classification_metrics(test_labels, predictions, probabilities)
            signals = pd.Series(predictions, index=split.test.index).map({0: "SELL", 1: "HOLD", 2: "BUY"})
            equity = backtest(split.test["close"], signals)["equity"]
            strategy = strategy_metrics(equity)
            rows.append({"fold": fold, "model": model_name, **classification, **strategy})
        fold += 1
        train_end += test_size
    if not rows:
        raise ValueError("資料不足以進行 walk-forward 評估")
    return pd.DataFrame(rows)
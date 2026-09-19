"""Logistic Regression 與 Random Forest 基準模型。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .labels import encode_labels


@dataclass
class TemporalSplit:
    """保存依時間順序切分的資料。"""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_split(frame: pd.DataFrame, train_ratio: float = 0.7, validation_ratio: float = 0.15) -> TemporalSplit:
    """依時間順序切分，禁止 random split。"""
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("切分比例不合法")
    train_end = int(len(frame) * train_ratio)
    validation_end = train_end + int(len(frame) * validation_ratio)
    if train_end < 1 or validation_end >= len(frame):
        raise ValueError("資料不足以進行時間切分")
    return TemporalSplit(frame.iloc[:train_end], frame.iloc[train_end:validation_end], frame.iloc[validation_end:])


def fit_baselines(
    split: TemporalSplit, feature_names: list[str], random_state: int = 42, n_estimators: int = 100
) -> tuple[dict[str, object], StandardScaler]:
    """只用 training fit scaler 與兩個分類器。"""
    scaler = StandardScaler().fit(split.train[feature_names])
    x_train = scaler.transform(split.train[feature_names])
    y_train = encode_labels(split.train["label"])
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=500, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=n_estimators, random_state=random_state),
    }
    for model in models.values():
        model.fit(x_train, y_train)
    return models, scaler


def transform_split(frame: pd.DataFrame, feature_names: list[str], scaler: StandardScaler) -> tuple[pd.DataFrame, pd.Series]:
    """使用已由 training fit 的 scaler 轉換資料。"""
    return pd.DataFrame(scaler.transform(frame[feature_names]), index=frame.index, columns=feature_names), encode_labels(frame["label"])

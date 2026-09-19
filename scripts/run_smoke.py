from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import backtest
from src.data_loader import generate_mock_ohlcv
from src.features import add_features, feature_columns
from src.labels import ID_TO_LABEL, add_direction_labels
from src.lstm import LSTMClassifier, make_sequences, train_lstm
from src.metrics import classification_metrics, strategy_metrics
from src.ml_baseline import fit_baselines, temporal_split, transform_split


def main() -> None:
    with (ROOT / "config" / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    torch.manual_seed(config["project"]["random_seed"])
    frame = add_features(
        generate_mock_ohlcv(config["data"]["mock_rows"], config["project"]["random_seed"]),
        **config["features"],
    )
    frame = add_direction_labels(
        frame,
        horizon=config["data"]["horizon_bars"],
        threshold=config["data"]["label_threshold"],
    )
    split = temporal_split(
        frame,
        train_ratio=config["split"]["train_ratio"],
        validation_ratio=config["split"]["validation_ratio"],
    )
    names = feature_columns(frame)
    models, scaler = fit_baselines(
        split,
        names,
        random_state=config["project"]["random_seed"],
        n_estimators=config["models"]["random_forest_estimators"],
    )
    test_features, test_labels = transform_split(split.test, names, scaler)
    print(f"rows={len(frame)} train={len(split.train)} validation={len(split.validation)} test={len(split.test)}")
    for name, model in models.items():
        predictions = model.predict(test_features)
        probabilities = model.predict_proba(test_features)
        print(f"{name} metrics={classification_metrics(test_labels, predictions, probabilities)}")

    train_features, train_labels = transform_split(split.train, names, scaler)
    sequence_length = config["split"]["sequence_length"]
    train_x, train_y = make_sequences(
        train_features.to_numpy(dtype=np.float32), train_labels.to_numpy(), sequence_length
    )
    test_x, test_y = make_sequences(
        test_features.to_numpy(dtype=np.float32), test_labels.to_numpy(), sequence_length
    )
    lstm = LSTMClassifier(
        input_size=len(names),
        hidden_size=config["models"]["lstm_hidden_size"],
        num_layers=config["models"]["lstm_layers"],
    )
    train_lstm(
        lstm,
        train_x,
        train_y,
        epochs=config["models"]["epochs"],
        learning_rate=config["models"]["learning_rate"],
        batch_size=config["models"]["batch_size"],
    )
    lstm.eval()
    with torch.no_grad():
        lstm_predictions = lstm(torch.as_tensor(test_x, dtype=torch.float32)).argmax(dim=1).numpy()
    print(f"lstm metrics={classification_metrics(test_y, lstm_predictions)}")

    prices = split.test["close"].iloc[sequence_length:]
    signals = pd.Series(
        [ID_TO_LABEL[int(label)] for label in lstm_predictions], index=prices.index
    )
    result = backtest(prices, signals)
    print(f"backtest metrics={strategy_metrics(result['equity'])}")


if __name__ == "__main__":
    main()
"""PyTorch LSTM 三分類器與時間序列樣本建立工具。"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # 允許未安裝 PyTorch 時先匯入其餘模組
    torch = None
    nn = None


def make_sequences(features: np.ndarray, labels: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """以過去 sequence_length 根 K 棒建立 LSTM 輸入。"""
    if sequence_length < 1 or len(features) <= sequence_length:
        raise ValueError("資料不足以建立序列")
    x = np.stack([features[index - sequence_length:index] for index in range(sequence_length, len(features))])
    return x, labels[sequence_length:]


def train_lstm(
    model: "LSTMClassifier",
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    learning_rate: float = 0.001,
    batch_size: int = 32,
) -> "LSTMClassifier":
    """在 CPU 上以 mini-batch 訓練 LSTM。"""
    if torch is None:
        raise ImportError("使用 train_lstm 前請安裝 PyTorch")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs 與 batch_size 必須為正數")
    device = torch.device("cpu")
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_features.to(device))
            loss = criterion(outputs, batch_labels.to(device))
            loss.backward()
            optimizer.step()
    return model


if nn is not None:

    class LSTMClassifier(nn.Module):
        """最小 LSTM 分類器。"""

        def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, classes: int = 3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.classifier = nn.Linear(hidden_size, classes)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs, _ = self.lstm(inputs)
            return self.classifier(outputs[:, -1, :])

else:

    class LSTMClassifier:  # type: ignore[no-redef]
        """未安裝 PyTorch 時提供清楚的錯誤訊息。"""

        def __init__(self, *_args, **_kwargs):
            raise ImportError("使用 LSTMClassifier 前請安裝 PyTorch")

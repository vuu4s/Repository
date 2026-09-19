from src.data_loader import generate_mock_ohlcv
from src.labels import LABEL_TO_ID, add_direction_labels, encode_labels


def test_labels_use_three_classes_and_drop_unknown_future_rows():
    labels = add_direction_labels(generate_mock_ohlcv(100), horizon=5, threshold=0.0)
    encoded = encode_labels(labels["label"])
    assert set(labels["label"]).issubset(LABEL_TO_ID)
    assert set(encoded).issubset({0, 1, 2})
    assert len(labels) == 95

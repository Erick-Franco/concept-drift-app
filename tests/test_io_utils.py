from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import (
    detect_label_column,
    encode_features,
    load_dataset,
    load_drift_ground_truth,
    list_builtin_datasets,
)


def test_detect_label_column_finds_named_column():
    df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
    assert detect_label_column(df) == "target"


def test_detect_label_column_returns_none_for_all_numeric_features():
    df = pd.DataFrame({"a": [1.1, 2.2, 3.3], "b": [4.4, 5.5, 6.6]})
    assert detect_label_column(df) is None


def test_encode_features_one_hot_encodes_categoricals_and_scales_numeric():
    df = pd.DataFrame(
        {
            "protocol_type": ["tcp", "udp", "tcp", "tcp"] * 20,
            "num_col": list(range(80)),
            "label": [0, 1, 0, 1] * 20,
        }
    )
    X, y, names = encode_features(df, "label")
    assert X.shape[0] == 80
    assert X.dtype == np.float32
    assert y.tolist() == df["label"].tolist()
    assert "label" not in names


def test_encode_features_handles_unseen_category_without_crashing():
    train = pd.DataFrame({"service": ["http", "ftp"] * 10, "label": [0, 1] * 10})
    X, y, names = encode_features(train, "label")
    assert X.shape[0] == 20


def test_load_drift_ground_truth_matches_known_dataset():
    path = Path("datos_sinteticos/sinteticos/sea_abrupt_seed1.csv")
    assert load_drift_ground_truth(path) == [5000, 9000]


def test_load_drift_ground_truth_returns_none_for_unknown_dataset():
    path = Path("data/data/sintetico_hyperplane_40f_drift50k.csv")
    assert load_drift_ground_truth(path) is None


def test_list_builtin_datasets_finds_known_files():
    datasets = list_builtin_datasets()
    assert any("sea_abrupt_seed1" in k for k in datasets)
    assert all(isinstance(v, Path) for v in datasets.values())


def test_load_dataset_reads_csv():
    df = load_dataset(Path("datos_sinteticos/sinteticos/sea_abrupt_seed1.csv"))
    assert list(df.columns) == ["t", "x0", "x1", "x2", "y"]

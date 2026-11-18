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


def test_detect_label_column_does_not_mistake_a_small_int_feature_for_a_label():
    df = pd.DataFrame({"a": [1.1, 2.2, 3.3, 4.4], "count": [0, 1, 2, 4]})
    assert detect_label_column(df) is None


def test_detect_label_column_accepts_unnamed_binary_last_column():
    df = pd.DataFrame({"a": [1.1, 2.2, 3.3, 4.4], "flag": [0, 1, 0, 1]})
    assert detect_label_column(df) == "flag"


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
    rng = np.random.default_rng(0)
    fit_rows = pd.DataFrame({"service": ["http", "ftp"], "label": [0, 1]})
    fit_rows = pd.concat([fit_rows] * 30, ignore_index=True)  # 60 rows, only 2 categories in fit slice
    extra_rows = pd.DataFrame({"service": ["telnet", "smtp"], "label": [0, 1]})
    df = pd.concat([fit_rows, extra_rows], ignore_index=True)
    X, y, names = encode_features(df, "label")
    assert X.shape[0] == 62


def test_encode_features_handles_high_cardinality_categorical_without_crashing():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "service": [f"svc_{i % 40}" for i in range(n)],
            "num_col": rng.normal(size=n),
            "label": rng.integers(0, 2, size=n),
        }
    )
    X, y, names = encode_features(df, "label")
    assert X.shape[0] == n
    assert isinstance(X, np.ndarray)


def test_encode_features_drops_trivial_row_index_column():
    n = 100
    df = pd.DataFrame(
        {
            "t": list(range(n)),
            "x0": np.random.default_rng(0).normal(size=n),
            "label": [0, 1] * (n // 2),
        }
    )
    X, y, names = encode_features(df, "label")
    assert "t" not in names
    assert X.shape[1] == 1


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

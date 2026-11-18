"""Carga de datasets, deteccion de etiqueta y codificacion de features."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BUILTIN_DATASET_DIRS = [Path("data/data"), Path("datos_sinteticos/sinteticos")]
DRIFTS_JSON_PATH = Path("datos_sinteticos/sinteticos/drifts.json")
LABEL_COLUMN_CANDIDATES = {"label", "target", "y", "class"}
MAX_LABEL_UNIQUE_VALUES = 10
MIN_SCALER_FIT_ROWS = 50
SCALER_FIT_FRACTION = 0.2


def list_builtin_datasets() -> dict[str, Path]:
    datasets: dict[str, Path] = {}
    for directory in BUILTIN_DATASET_DIRS:
        if not directory.exists():
            continue
        for csv_path in sorted(directory.glob("*.csv")):
            datasets[csv_path.stem] = csv_path
    return datasets


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def detect_label_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        if column.strip().lower() in LABEL_COLUMN_CANDIDATES:
            return column
    last_column = df.columns[-1]
    if pd.api.types.is_float_dtype(df[last_column]):
        return None
    if df[last_column].nunique() <= MAX_LABEL_UNIQUE_VALUES:
        return last_column
    return None


def encode_features(
    df: pd.DataFrame, label_col: str | None
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    feature_df = df.drop(columns=[label_col]) if label_col else df
    y = df[label_col].to_numpy() if label_col else None

    categorical_cols = feature_df.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

    fit_rows = max(MIN_SCALER_FIT_ROWS, int(len(feature_df) * SCALER_FIT_FRACTION))
    fit_rows = min(fit_rows, len(feature_df))
    fit_slice = feature_df.iloc[:fit_rows]

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    transformer.fit(fit_slice)
    X = transformer.transform(feature_df)
    X = np.asarray(X, dtype=np.float32)

    feature_names = list(transformer.get_feature_names_out())
    return X, y, feature_names


def load_drift_ground_truth(dataset_path: Path) -> list[int] | None:
    if not DRIFTS_JSON_PATH.exists():
        return None
    with DRIFTS_JSON_PATH.open() as f:
        drifts = json.load(f)
    stem = dataset_path.stem
    for key, entry in drifts.items():
        if stem.startswith(key):
            return entry["ground_truth_inicio"]
    return None

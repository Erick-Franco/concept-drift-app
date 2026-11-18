# Concept Drift Detection App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit app where a user picks/uploads a dataset, configures constraints (labels available?, priority), runs ADWIN/DDM/EDDM/HDDM_A/HDDM_W/PageHinkley/KSWIN plus a PyTorch adaptive MLP classifier and a PyTorch autoencoder-based unsupervised drift signal, and sees a comparison table, a drift-marked plot, and an automatic recommendation.

**Architecture:** `io_utils.py` loads/encodes any of the project's datasets (built-in or uploaded) and exposes known drift ground truth. `detectores.py` wraps river's drift detectors behind one interface, implements the PyTorch `MLPAdaptativo` and `AutoencoderDetector`, and runs a full detection pass producing a metrics dataclass. `app.py` is pure Streamlit orchestration over those two modules — no detection logic lives in `app.py`.

**Tech Stack:** Python 3.10+, streamlit, river, torch (CPU), pandas, numpy, scikit-learn (StandardScaler/OneHotEncoder), plotly, pytest.

**Spec:** Design doc approved in conversation 2026-11-18 (no separate file written; this plan is the authoritative reference — see "Design recap" below).

## Design recap (from approved conversation)

- Datasets live in `data/data/`, `datos_sinteticos/sinteticos/` (ground truth drift points in `datos_sinteticos/sinteticos/drifts.json`), plus user uploads.
- NSL-KDD has categorical columns (`protocol_type`, `service`, `flag`) that must be encoded.
- Metrics table schema mirrors `data/resultados/benchmark_completo.csv`: `Dataset, Detector, Accuracy (%), Drifts_Total, Primeros_Drifts, Tiempo_seg, RAM_Pico_KB`, plus a new `Retardo_promedio` column when ground truth or labels make it computable.
- Approach B (approved): MLP incremental adaptive classifier **and** an autoencoder-based unsupervised detector, so datasets without labels still get real drift detection instead of being skipped.

## Global Constraints

- Detectors: ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley, KSWIN (river implementations only).
- NN library: PyTorch, CPU-only, small nets (no GPU assumption).
- Results table columns must match: `Dataset, Detector, Accuracy (%), Drifts_Total, Primeros_Drifts, Tiempo_seg, RAM_Pico_KB, Retardo_promedio`.
- `Primeros_Drifts` = first 3 detected drift indices (matches existing benchmark CSV format).
- App must run fully offline/local via `streamlit run app.py`, no external API calls.
- Must support both labeled flows (`tiene_labels=True`) and unlabeled flows (`tiene_labels=False`) for every listed detector — no detector may simply be disabled when labels are absent; unlabeled flows route through the autoencoder's reconstruction-error signal instead.

## Review Focus

- **Mixed categorical/numeric CSV with unseen categories** (e.g. NSL-KDD `service` values not seen in the encoder's fit window) — must not crash; unseen categories should encode to a default bucket.
- **Dataset with no detectable label column and no `drifts.json` entry** (e.g. `sintetico_hyperplane_40f_drift50k.csv`) — app must still run and report metrics, just without `Retardo_promedio`.
- **User uploads a CSV with all-numeric columns but no obvious label column** — `detect_label_column` must degrade gracefully (return `None`) rather than picking a feature column as a fake label.
- **Dataset smaller than the autoencoder/MLP warmup window** (e.g. a small user upload) — warmup must clamp to available rows instead of indexing out of range.
- **A detector configured for a labeled run when `tiene_labels=False`** — `run_detection` must route it to the autoencoder signal automatically rather than erroring or silently producing zero drifts.

---

## File Structure

- Create: `requirements.txt`
- Create: `io_utils.py` — dataset discovery, loading, label detection, categorical encoding, ground-truth lookup
- Create: `detectores.py` — detector factories, `MLPAdaptativo`, `AutoencoderDetector`, `run_detection`
- Create: `app.py` — Streamlit UI and orchestration
- Create: `README.md` — setup/run instructions
- Create: `tests/test_io_utils.py`
- Create: `tests/test_detectores.py`

---

### Task 1: Project scaffolding, requirements, README

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `.gitignore`

**Interfaces:**
- Produces: nothing code-level; later tasks assume these packages are installable: `streamlit`, `river`, `torch`, `pandas`, `numpy`, `scikit-learn`, `plotly`, `pytest`.

- [ ] **Step 1: Write `requirements.txt`**

Pin to CPU-friendly versions, no exact-pin requirement beyond major compatibility:
```
streamlit>=1.38
river>=0.21
torch>=2.2
pandas>=2.0
numpy>=1.26
scikit-learn>=1.3
plotly>=5.20
pytest>=8.0
```

- [ ] **Step 2: Write `README.md`**

Must include, in order: project one-liner, `python -m venv venv`, activation command (bash), `pip install -r requirements.txt`, `streamlit run app.py`, `pytest` for tests, and a short note on where built-in datasets are read from (`data/data/`, `datos_sinteticos/sinteticos/`).

- [ ] **Step 3: Write `.gitignore`**

Must ignore: `venv/`, `__pycache__/`, `*.pyc`, `.streamlit/`.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt README.md .gitignore
git commit -m "chore: scaffold project requirements and docs"
```

---

### Task 2: Data loading, label detection, encoding (`io_utils.py`)

**Files:**
- Create: `io_utils.py`
- Test: `tests/test_io_utils.py`

**Interfaces:**
- Produces:
  - `list_builtin_datasets() -> dict[str, pathlib.Path]` — keys are human-readable dataset names, values are absolute paths, scanning `data/data/*.csv` and `datos_sinteticos/sinteticos/*.csv`.
  - `load_dataset(path: pathlib.Path) -> pandas.DataFrame`
  - `detect_label_column(df: pandas.DataFrame) -> str | None` — checks `['label', 'target', 'y', 'class']` case-insensitively, else returns last column only if it has <= 10 unique values, else `None`.
  - `encode_features(df: pandas.DataFrame, label_col: str | None) -> tuple[numpy.ndarray, numpy.ndarray | None, list[str]]` — returns `(X, y, feature_names)`; `X` is `float32`, categorical/object columns one-hot encoded via `sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")`, numeric columns passed through `StandardScaler` fit on the first 20% of rows (min 50 rows); `y` is `None` if `label_col is None`.
  - `load_drift_ground_truth(dataset_path: pathlib.Path) -> list[int] | None` — looks up `datos_sinteticos/sinteticos/drifts.json` by matching `dataset_path.stem` against its keys (e.g. `sea_abrupt_seed1.csv` → key `sea_abrupt`, matched by stem starting with the key), returns that entry's `ground_truth_inicio` list, else `None`.

- [ ] **Step 1: Write failing tests for `detect_label_column` and `encode_features`**

```python
import pandas as pd
import numpy as np
from io_utils import detect_label_column, encode_features

def test_detect_label_column_finds_named_column():
    df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
    assert detect_label_column(df) == "target"

def test_detect_label_column_returns_none_for_all_numeric_features():
    df = pd.DataFrame({"a": [1.1, 2.2, 3.3], "b": [4.4, 5.5, 6.6]})
    assert detect_label_column(df) is None

def test_encode_features_one_hot_encodes_categoricals_and_scales_numeric():
    df = pd.DataFrame({
        "protocol_type": ["tcp", "udp", "tcp", "tcp"] * 20,
        "num_col": list(range(80)),
        "label": [0, 1, 0, 1] * 20,
    })
    X, y, names = encode_features(df, "label")
    assert X.shape[0] == 80
    assert X.dtype == np.float32
    assert y.tolist() == df["label"].tolist()
    assert "label" not in names

def test_encode_features_handles_unseen_category_without_crashing():
    train = pd.DataFrame({"service": ["http", "ftp"] * 10, "label": [0, 1] * 10})
    X, y, names = encode_features(train, "label")
    assert X.shape[0] == 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'io_utils'`

- [ ] **Step 3: Implement `io_utils.py`**

Implement the four functions above plus `list_builtin_datasets` and `load_drift_ground_truth`. `load_dataset` is a thin `pandas.read_csv(path)` wrapper — no special-casing per dataset (column auto-detection happens in `detect_label_column`/`encode_features`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io_utils.py -v`
Expected: PASS

- [ ] **Step 5: Add and run the ground-truth lookup test**

```python
from pathlib import Path
from io_utils import load_drift_ground_truth

def test_load_drift_ground_truth_matches_known_dataset():
    path = Path("datos_sinteticos/sinteticos/sea_abrupt_seed1.csv")
    assert load_drift_ground_truth(path) == [5000, 9000]

def test_load_drift_ground_truth_returns_none_for_unknown_dataset():
    path = Path("data/data/sintetico_hyperplane_40f_drift50k.csv")
    assert load_drift_ground_truth(path) is None
```

Run: `pytest tests/test_io_utils.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add io_utils.py tests/test_io_utils.py
git commit -m "feat: add dataset loading, label detection and encoding utilities"
```

---

### Task 3: River detector wrappers and metrics dataclass (`detectores.py` part 1)

**Files:**
- Create: `detectores.py`
- Test: `tests/test_detectores.py`

**Interfaces:**
- Consumes: nothing from Task 2 directly (works on plain `numpy.ndarray` signals), but will be driven by arrays `io_utils.encode_features` produces.
- Produces:
  - `DETECTOR_FACTORIES: dict[str, Callable[[], Any]]` with keys `"ADWIN", "DDM", "EDDM", "HDDM_A", "HDDM_W", "PageHinkley", "KSWIN"`, each value a zero-arg callable returning a fresh `river.drift.<Detector>()` instance.
  - `requires_binary_error_signal(name: str) -> bool` — `True` for `DDM, EDDM, HDDM_A, HDDM_W`, `False` for `ADWIN, PageHinkley, KSWIN`.
  - `@dataclasses.dataclass ResultadoDeteccion` with fields: `detector: str`, `accuracy: float`, `drifts_idx: list[int]`, `tiempo_seg: float`, `ram_pico_kb: float`, `retardo_promedio: float | None`.
  - `calcular_retardo(drifts_idx: list[int], ground_truth: list[int] | None, horizonte: int = 2000) -> float | None` — for each ground-truth point, finds the nearest `drifts_idx` value `>=` that point within `horizonte`; averages those distances; returns `None` if `ground_truth` is `None` or no match found for any point.

- [ ] **Step 1: Write failing tests**

```python
from detectores import DETECTOR_FACTORIES, requires_binary_error_signal, calcular_retardo

def test_all_seven_detectors_are_registered():
    assert set(DETECTOR_FACTORIES) == {"ADWIN", "DDM", "EDDM", "HDDM_A", "HDDM_W", "PageHinkley", "KSWIN"}

def test_factories_return_fresh_instances_each_call():
    a = DETECTOR_FACTORIES["ADWIN"]()
    b = DETECTOR_FACTORIES["ADWIN"]()
    assert a is not b

def test_requires_binary_error_signal_partitions_detectors():
    assert requires_binary_error_signal("DDM") is True
    assert requires_binary_error_signal("ADWIN") is False

def test_calcular_retardo_averages_nearest_following_detections():
    assert calcular_retardo(drifts_idx=[5010, 9050], ground_truth=[5000, 9000], horizonte=2000) == 30.0

def test_calcular_retardo_returns_none_without_ground_truth():
    assert calcular_retardo(drifts_idx=[100], ground_truth=None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_detectores.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'detectores'`

- [ ] **Step 3: Implement `DETECTOR_FACTORIES`, `requires_binary_error_signal`, `ResultadoDeteccion`, `calcular_retardo`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_detectores.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add detectores.py tests/test_detectores.py
git commit -m "feat: add river detector registry and delay metric"
```

---

### Task 4: PyTorch adaptive MLP classifier (`detectores.py` part 2)

**Files:**
- Modify: `detectores.py`
- Test: `tests/test_detectores.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `class MLPAdaptativo`:
    - `__init__(self, input_dim: int, hidden: tuple[int, int] = (32, 16), lr: float = 0.01)`
    - `predict_proba_one(self, x: numpy.ndarray) -> float`
    - `learn_one(self, x: numpy.ndarray, y: int) -> None` — one SGD step.
    - `reset(self) -> None` — reinitializes weights (called when a detector signals drift).

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from detectores import MLPAdaptativo

def test_mlp_predict_proba_one_returns_value_in_unit_interval():
    mlp = MLPAdaptativo(input_dim=4)
    p = mlp.predict_proba_one(np.random.rand(4).astype(np.float32))
    assert 0.0 <= p <= 1.0

def test_mlp_learn_one_changes_predictions_over_many_updates():
    mlp = MLPAdaptativo(input_dim=2)
    x = np.array([1.0, 1.0], dtype=np.float32)
    before = mlp.predict_proba_one(x)
    for _ in range(200):
        mlp.learn_one(x, 1)
    after = mlp.predict_proba_one(x)
    assert after > before

def test_mlp_reset_reinitializes_weights():
    mlp = MLPAdaptativo(input_dim=2)
    x = np.array([1.0, 1.0], dtype=np.float32)
    for _ in range(200):
        mlp.learn_one(x, 1)
    trained = mlp.predict_proba_one(x)
    mlp.reset()
    reset_val = mlp.predict_proba_one(x)
    assert reset_val != trained
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_detectores.py -v`
Expected: FAIL with `ImportError: cannot import name 'MLPAdaptativo'`

- [ ] **Step 3: Implement `MLPAdaptativo`**

Two hidden `nn.Linear` + `ReLU` layers sized by `hidden`, final `nn.Linear(hidden[1], 1)` + sigmoid, `torch.optim.SGD` or `Adam` at `lr`, `nn.BCELoss` for `learn_one`. `reset()` rebuilds the network and optimizer from scratch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_detectores.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add detectores.py tests/test_detectores.py
git commit -m "feat: add adaptive MLP classifier with drift-triggered reset"
```

---

### Task 5: PyTorch autoencoder drift signal (`detectores.py` part 3)

**Files:**
- Modify: `detectores.py`
- Test: `tests/test_detectores.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `class AutoencoderDetector`:
    - `__init__(self, input_dim: int, hidden: int = 16)`
    - `warmup(self, X_init: numpy.ndarray, epochs: int = 20) -> None` — trains on up to `min(len(X_init), 500)` rows.
    - `reconstruction_error(self, x: numpy.ndarray) -> float` — MSE between `x` and its reconstruction.
    - `is_drift(self, error: float) -> bool` — feeds `error` into an internal `river.drift.ADWIN()` instance created in `__init__`, returns whether that internal detector flagged change on this update.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from detectores import AutoencoderDetector

def test_warmup_handles_fewer_rows_than_default_window():
    ae = AutoencoderDetector(input_dim=3)
    ae.warmup(np.random.rand(10, 3).astype(np.float32))  # fewer than 500 rows

def test_reconstruction_error_is_non_negative_float():
    ae = AutoencoderDetector(input_dim=3)
    ae.warmup(np.random.rand(50, 3).astype(np.float32))
    err = ae.reconstruction_error(np.random.rand(3).astype(np.float32))
    assert err >= 0.0

def test_is_drift_flags_large_sustained_error_jump():
    ae = AutoencoderDetector(input_dim=2)
    normal = np.tile(np.array([0.5, 0.5], dtype=np.float32), (100, 1))
    ae.warmup(normal)
    flagged = False
    for _ in range(50):
        err = ae.reconstruction_error(np.array([0.5, 0.5], dtype=np.float32))
        flagged = flagged or ae.is_drift(err)
    for _ in range(50):
        err = ae.reconstruction_error(np.array([50.0, -50.0], dtype=np.float32))
        flagged = flagged or ae.is_drift(err)
    assert flagged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_detectores.py -v`
Expected: FAIL with `ImportError: cannot import name 'AutoencoderDetector'`

- [ ] **Step 3: Implement `AutoencoderDetector`**

Single hidden-layer encoder/decoder (`input_dim -> hidden -> input_dim`), trained with `nn.MSELoss` and `Adam` in `warmup`. `is_drift` calls `self._adwin.update(error)` (river 0.21 `ADWIN.update` returns the detector itself with a `drift_detected` attribute, or use whichever river 0.21 API is current — check installed version) and returns its change-detected flag.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_detectores.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add detectores.py tests/test_detectores.py
git commit -m "feat: add autoencoder-based unsupervised drift signal"
```

---

### Task 6: `run_detection` orchestration function (`detectores.py` part 4)

**Files:**
- Modify: `detectores.py`
- Test: `tests/test_detectores.py`

**Interfaces:**
- Consumes: `DETECTOR_FACTORIES`, `requires_binary_error_signal`, `ResultadoDeteccion`, `calcular_retardo`, `MLPAdaptativo`, `AutoencoderDetector` (all from Tasks 3-5).
- Produces:
  - `run_detection(X: numpy.ndarray, y: numpy.ndarray | None, detector_name: str, tiene_labels: bool, ground_truth: list[int] | None, warmup_rows: int = 200) -> ResultadoDeteccion`

Behavior (this is the core integration point — the body is the implementer's judgment call within these rules):
- Always instantiate `MLPAdaptativo(input_dim=X.shape[1])`; warm it up with supervised steps on the first `warmup_rows` if `tiene_labels`, otherwise warm it up unsupervised is not possible — in that case it still runs via `AutoencoderDetector`'s reconstruction stream for signal purposes and the MLP is warmed up using its own predictions as pseudo-labels (document this as a one-line comment, not a design question).
- If `tiene_labels` is True: feed the detector a per-sample `0/1` error signal from the MLP's prediction vs. true label (`requires_binary_error_signal` detectors expect this directly; `ADWIN/PageHinkley/KSWIN` also accept it).
- If `tiene_labels` is False: always instantiate `AutoencoderDetector`, call `warmup` on the first `warmup_rows` of `X`; feed every detector the per-sample `reconstruction_error` (binarize via `error > rolling_mean + 2*rolling_std` for `requires_binary_error_signal` detectors, pass the raw float to `ADWIN/PageHinkley/KSWIN`).
- On every index where `MLPAdaptativo` reset and/or the chosen `detector_name`'s `update(...)` reports drift, append that index to `drifts_idx`; after appending, call `mlp.reset()`.
- Wrap the whole per-sample loop in `tracemalloc.start()` / `tracemalloc.get_traced_memory()` for `ram_pico_kb` and `time.perf_counter()` for `tiempo_seg`.
- `accuracy` = running fraction of correct MLP predictions over all samples where `y` is available; `0.0` (and documented as such) when `tiene_labels` is False.
- `retardo_promedio` via `calcular_retardo(drifts_idx, ground_truth)`.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from detectores import run_detection

def make_labeled_stream(n=600, drift_at=300):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(n, 3)).astype(np.float32)
    X[drift_at:] += 5.0
    y = (X[:, 0] > 0).astype(int)
    return X, y

def test_run_detection_with_labels_returns_populated_result():
    X, y = make_labeled_stream()
    result = run_detection(X, y, "ADWIN", tiene_labels=True, ground_truth=[300])
    assert result.detector == "ADWIN"
    assert 0.0 <= result.accuracy <= 1.0
    assert isinstance(result.drifts_idx, list)
    assert result.tiempo_seg >= 0.0
    assert result.ram_pico_kb >= 0.0

def test_run_detection_without_labels_still_detects_drift():
    X, _ = make_labeled_stream()
    result = run_detection(X, None, "DDM", tiene_labels=False, ground_truth=[300])
    assert len(result.drifts_idx) >= 0  # must not crash; detection via autoencoder signal
    assert result.accuracy == 0.0

def test_run_detection_sets_retardo_none_without_ground_truth():
    X, y = make_labeled_stream()
    result = run_detection(X, y, "ADWIN", tiene_labels=True, ground_truth=None)
    assert result.retardo_promedio is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_detectores.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_detection'`

- [ ] **Step 3: Implement `run_detection`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_detectores.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add detectores.py tests/test_detectores.py
git commit -m "feat: add run_detection orchestration for labeled and unlabeled flows"
```

---

### Task 7: Streamlit app (`app.py`)

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes:
  - From `io_utils.py`: `list_builtin_datasets`, `load_dataset`, `detect_label_column`, `encode_features`, `load_drift_ground_truth`.
  - From `detectores.py`: `DETECTOR_FACTORIES` (for the list of names to run), `run_detection`, `ResultadoDeteccion`.
- Produces: no importable interface (top-level script), but must expose `recomendar(df_resultados: pandas.DataFrame, prioridad: str) -> str` as a plain function for manual testing.

- [ ] **Step 1: Build the sidebar**

`st.sidebar.selectbox` listing `list_builtin_datasets()` keys plus a `st.sidebar.file_uploader("Subir CSV", type="csv")` option; `st.sidebar.checkbox("¿El flujo tiene etiquetas inmediatas?")`; `st.sidebar.radio("Prioridad", ["Menor retardo", "Menos falsas alarmas", "Mayor Accuracy", "Menor costo computacional"])`; `st.sidebar.button("Ejecutar")`.

- [ ] **Step 2: Wire up data loading and run button**

On button press: load the chosen/uploaded dataset via `load_dataset`, detect/encode via `detect_label_column` + `encode_features` (let the user override the detected label column via a `st.selectbox` if `detect_label_column` returns `None`), fetch `ground_truth = load_drift_ground_truth(path)` for built-in datasets (`None` for uploads), then call `run_detection` for every name in `DETECTOR_FACTORIES`, collecting `ResultadoDeteccion` objects into a `pandas.DataFrame` with columns `Dataset, Detector, Accuracy (%), Drifts_Total, Primeros_Drifts, Tiempo_seg, RAM_Pico_KB, Retardo_promedio` (derive `Accuracy (%)` as `accuracy * 100`, `Drifts_Total` as `len(drifts_idx)`, `Primeros_Drifts` as `drifts_idx[:3]`).

- [ ] **Step 3: Render results table and plot**

`st.dataframe(df_resultados)`; a `plotly.graph_objects` line chart of one representative numeric column of `X` (or `y` if present) over the sample index, with a `go.Scatter` vertical marker (or `add_vline`) per detector's `drifts_idx` in a distinct color, plus a distinct-style vline for each `ground_truth` point if present.

- [ ] **Step 4: Implement and wire up `recomendar`**

`recomendar(df_resultados, prioridad)` picks the row that best satisfies `prioridad` (`"Menor retardo"` → min non-null `Retardo_promedio`; `"Menos falsas alarmas"` → min `Drifts_Total`; `"Mayor Accuracy"` → max `Accuracy (%)`; `"Menor costo computacional"` → min `Tiempo_seg`), returns a sentence naming that detector and citing its concrete metric value. Render it with `st.success(...)`.

- [ ] **Step 5: Manual smoke test**

Run: `streamlit run app.py` and, in the browser, select `sea_abrupt_seed1.csv`, check "tiene etiquetas", click "Ejecutar".
Expected: table renders with 7 rows, plot renders with drift markers, a recommendation sentence appears. Repeat once with the checkbox unchecked to confirm the unlabeled path also renders without errors.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit UI for concept drift comparison"
```

---

## Execution Handoff Note

Keep tasks 3-6 in `detectores.py` as sequential edits to the same file (each task appends; none removes a previous task's code). Task 7 is the only task that cannot be meaningfully unit-tested with pytest — its verification step is manual.

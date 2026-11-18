"""Deteccion de concept drift: detectores de river, MLP adaptativo y autoencoder."""

import dataclasses
import time
import tracemalloc
from typing import Callable

import numpy as np
import torch
from torch import nn
from river.drift import KSWIN, PageHinkley
from river.drift import ADWIN as RiverADWIN
from river.drift.binary import DDM, EDDM, HDDMA, HDDMW

DETECTOR_FACTORIES: dict[str, Callable[[], object]] = {
    "ADWIN": lambda: RiverADWIN(),
    "DDM": lambda: DDM(),
    "EDDM": lambda: EDDM(),
    "HDDM_A": lambda: HDDMA(),
    "HDDM_W": lambda: HDDMW(),
    "PageHinkley": lambda: PageHinkley(),
    "KSWIN": lambda: KSWIN(),
}

_BINARY_SIGNAL_DETECTORS = {"DDM", "EDDM", "HDDM_A", "HDDM_W"}


def requires_binary_error_signal(name: str) -> bool:
    return name in _BINARY_SIGNAL_DETECTORS


@dataclasses.dataclass
class ResultadoDeteccion:
    detector: str
    accuracy: float
    drifts_idx: list[int]
    tiempo_seg: float
    ram_pico_kb: float
    retardo_promedio: float | None


def calcular_retardo(
    drifts_idx: list[int], ground_truth: list[int] | None, horizonte: int = 2000
) -> float | None:
    if not ground_truth:
        return None
    delays = []
    for gt_point in ground_truth:
        candidates = [idx for idx in drifts_idx if gt_point <= idx <= gt_point + horizonte]
        if candidates:
            delays.append(min(candidates) - gt_point)
    if not delays:
        return None
    return sum(delays) / len(delays)


class MLPAdaptativo:
    """Clasificador binario online: predice, aprende por muestra, y puede
    reiniciarse por completo cuando un detector de drift lo indica."""

    def __init__(self, input_dim: int, hidden: tuple[int, int] = (32, 16), lr: float = 0.01):
        self.input_dim = input_dim
        self.hidden = hidden
        self.lr = lr
        self._build()

    def _build(self) -> None:
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden[0]),
            nn.ReLU(),
            nn.Linear(self.hidden[0], self.hidden[1]),
            nn.ReLU(),
            nn.Linear(self.hidden[1], 1),
            nn.Sigmoid(),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.BCELoss()

    def predict_proba_one(self, x: np.ndarray) -> float:
        with torch.no_grad():
            tensor_x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
            return self.model(tensor_x).item()

    def learn_one(self, x: np.ndarray, y: int) -> None:
        tensor_x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
        tensor_y = torch.tensor([[float(y)]], dtype=torch.float32)
        self.optimizer.zero_grad()
        pred = self.model(tensor_x)
        loss = self.loss_fn(pred, tensor_y)
        loss.backward()
        self.optimizer.step()

    def reset(self) -> None:
        self._build()


class AutoencoderDetector:
    """Detector no supervisado: entrena un autoencoder sobre una ventana
    inicial y usa el error de reconstruccion, vigilado por un ADWIN interno,
    como senal de drift cuando no hay etiquetas disponibles."""

    MAX_WARMUP_ROWS = 500

    def __init__(self, input_dim: int, hidden: int = 16):
        self.input_dim = input_dim
        self.hidden = hidden
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )
        self._adwin = RiverADWIN()

    def warmup(self, X_init: np.ndarray, epochs: int = 20) -> None:
        n_rows = min(len(X_init), self.MAX_WARMUP_ROWS)
        tensor_x = torch.as_tensor(X_init[:n_rows], dtype=torch.float32)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            reconstruction = self.model(tensor_x)
            loss = loss_fn(reconstruction, tensor_x)
            loss.backward()
            optimizer.step()

    def reconstruction_error(self, x: np.ndarray) -> float:
        with torch.no_grad():
            tensor_x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
            reconstruction = self.model(tensor_x)
            return torch.mean((reconstruction - tensor_x) ** 2).item()

    def is_drift(self, error: float) -> bool:
        self._adwin.update(error)
        return self._adwin.drift_detected


def es_anomalia_binaria(error: float, baseline_mean: float, baseline_std: float) -> bool:
    """Decide si `error` es anomalo respecto a una linea base FIJA (calculada
    una sola vez durante el warmup). Una ventana deslizante se adaptaria al
    propio cambio sostenido y dejaria de verlo como anomalo; una linea base
    fija no."""
    return error > baseline_mean + 2 * baseline_std


def run_detection(
    X: np.ndarray,
    y: np.ndarray | None,
    detector_name: str,
    tiene_labels: bool,
    ground_truth: list[int] | None,
    warmup_rows: int = 200,
) -> ResultadoDeteccion:
    if tiene_labels and y is not None and len(np.unique(y)) > 2:
        raise ValueError(
            "MLPAdaptativo y los detectores binarios (DDM/EDDM/HDDM) solo soportan "
            "clasificacion binaria; la etiqueta seleccionada tiene mas de 2 clases."
        )

    torch.manual_seed(42)  # resultados deterministas y comparables entre detectores

    n_samples, n_features = X.shape
    mlp = MLPAdaptativo(input_dim=n_features)
    detector = DETECTOR_FACTORIES[detector_name]()
    binary_signal = requires_binary_error_signal(detector_name)

    autoencoder: AutoencoderDetector | None = None
    baseline_mean = baseline_std = 0.0
    if not tiene_labels:
        n_warmup = min(warmup_rows, n_samples)
        autoencoder = AutoencoderDetector(input_dim=n_features)
        autoencoder.warmup(X[:n_warmup])
        baseline_errors = [autoencoder.reconstruction_error(X[i]) for i in range(n_warmup)]
        baseline_mean = float(np.mean(baseline_errors))
        baseline_std = float(np.std(baseline_errors)) if len(baseline_errors) > 1 else 0.0
        # El MLP se mantiene activo con pseudo-etiquetas de sus propias
        # predicciones: no aporta accuracy (no hay etiquetas reales), pero
        # mantiene vivo el reset por drift para la arquitectura adaptativa.
        for i in range(n_warmup):
            pseudo_label = 1 if mlp.predict_proba_one(X[i]) > 0.5 else 0
            mlp.learn_one(X[i], pseudo_label)

    correct_count = 0
    total_labeled = 0
    drifts_idx: list[int] = []

    tracemalloc.start()
    t0 = time.perf_counter()

    for i in range(n_samples):
        x = X[i]

        if tiene_labels:
            true_label = int(y[i])
            pred_label = 1 if mlp.predict_proba_one(x) > 0.5 else 0
            correct = int(pred_label == true_label)
            correct_count += correct
            total_labeled += 1
            mlp.learn_one(x, true_label)
            signal = 1.0 - correct
        else:
            signal = autoencoder.reconstruction_error(x)

        if binary_signal:
            if tiene_labels:
                is_error = bool(signal)
            else:
                is_error = es_anomalia_binaria(signal, baseline_mean, baseline_std)
            detector.update(is_error)
        else:
            detector.update(float(signal))

        if getattr(detector, "drift_detected", False):
            drifts_idx.append(i)
            mlp.reset()

    tiempo_seg = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_pico_kb = peak_bytes / 1024

    accuracy = correct_count / total_labeled if total_labeled else 0.0
    retardo_promedio = calcular_retardo(drifts_idx, ground_truth)

    return ResultadoDeteccion(
        detector=detector_name,
        accuracy=accuracy,
        drifts_idx=drifts_idx,
        tiempo_seg=tiempo_seg,
        ram_pico_kb=ram_pico_kb,
        retardo_promedio=retardo_promedio,
    )

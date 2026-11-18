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

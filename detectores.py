"""Deteccion de concept drift: detectores de river, MLP adaptativo y autoencoder."""

import dataclasses
import time
import tracemalloc
from typing import Callable

import numpy as np
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

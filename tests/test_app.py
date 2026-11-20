import dataclasses

import numpy as np
import pandas as pd

from app import (
    _calcular_rango_zoom,
    _construir_tabla_resultados,
    _ejecutar_todos_los_detectores,
    _graficar_metricas_comparativas,
    _resaltar_fila_ganadora,
    _tabla_a_csv,
    contar_votos,
    elegir_ganador,
    obtener_metricas_ganador,
    recomendar,
)


@dataclasses.dataclass
class _ResultadoFalso:
    detector: str
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float
    drifts_idx: list
    tiempo_seg_mean: float
    ram_pico_kb_mean: float
    retardo_promedio_mean: float | None


def test_recomendar_menor_retardo_picks_min_delay():
    df = pd.DataFrame(
        {
            "Detector": ["ADWIN", "DDM"],
            "Accuracy (%)": [90.0, 95.0],
            "Drifts_Total": [2, 1],
            "Tiempo_seg": [1.0, 2.0],
            "Retardo_promedio": [50.0, 30.0],
        }
    )
    assert "DDM" in recomendar(df, "Menor retardo")


def test_recomendar_mayor_accuracy_is_not_misleading_without_labels():
    df = pd.DataFrame(
        {
            "Detector": ["ADWIN", "DDM"],
            "Accuracy (%)": [0.0, 0.0],
            "Drifts_Total": [2, 1],
            "Tiempo_seg": [1.0, 2.0],
            "Retardo_promedio": [None, None],
        }
    )
    mensaje = recomendar(df, "Mayor Accuracy", tiene_labels=False)
    assert "ADWIN" not in mensaje and "DDM" not in mensaje
    assert "sin etiquetas" in mensaje.lower() or "no se puede" in mensaje.lower()


def test_graficar_metricas_comparativas_includes_all_detectors():
    df = pd.DataFrame(
        {
            "Detector": ["ADWIN", "DDM", "EDDM"],
            "Accuracy (%)": [90.0, 85.0, 95.0],
            "Tiempo_seg": [1.0, 2.0, 0.5],
        }
    )
    fig = _graficar_metricas_comparativas(df)
    assert len(fig.data) == 2
    accuracy_trace = next(t for t in fig.data if t.name == "Accuracy (%)")
    tiempo_trace = next(t for t in fig.data if t.name == "Tiempo (s)")
    assert list(accuracy_trace.x) == ["ADWIN", "DDM", "EDDM"]
    assert list(accuracy_trace.y) == [90.0, 85.0, 95.0]
    assert list(tiempo_trace.y) == [1.0, 2.0, 0.5]


def test_construir_tabla_resultados_includes_f1_column():
    resultados = [
        _ResultadoFalso(
            detector="ADWIN", accuracy_mean=0.9, accuracy_std=0.01, f1_mean=0.8, f1_std=0.02,
            drifts_idx=[10], tiempo_seg_mean=1.0, ram_pico_kb_mean=100.0, retardo_promedio_mean=None,
        )
    ]
    tabla = _construir_tabla_resultados("mi_dataset", resultados)
    assert tabla.loc[0, "F1 (%)"] == 80.0
    assert tabla.loc[0, "Accuracy (%)"] == 90.0
    assert tabla.loc[0, "Accuracy_std (%)"] == 1.0
    assert tabla.loc[0, "F1_std (%)"] == 2.0


def test_elegir_ganador_menor_retardo_matches_recomendar():
    df = pd.DataFrame(
        {
            "Detector": ["ADWIN", "DDM"],
            "Accuracy (%)": [90.0, 95.0],
            "Drifts_Total": [2, 1],
            "Tiempo_seg": [1.0, 2.0],
            "Retardo_promedio": [50.0, 30.0],
        }
    )
    assert elegir_ganador(df, "Menor retardo", tiene_labels=True) == "DDM"


def test_elegir_ganador_returns_none_for_mayor_accuracy_without_labels():
    df = pd.DataFrame({"Detector": ["ADWIN"], "Accuracy (%)": [0.0]})
    assert elegir_ganador(df, "Mayor Accuracy", tiene_labels=False) is None


def test_contar_votos_tallies_winners_across_datasets():
    ganadores = {"dataset_a": "ADWIN", "dataset_b": "ADWIN", "dataset_c": "DDM", "dataset_d": None}
    votos = contar_votos(ganadores)
    assert votos["ADWIN"] == 2
    assert votos["DDM"] == 1
    assert "dataset_d" not in votos.index


def test_tabla_a_csv_round_trips_through_pandas():
    df = pd.DataFrame({"Detector": ["ADWIN", "DDM"], "Accuracy (%)": [90.0, 85.0]})
    csv_bytes = _tabla_a_csv(df)
    assert isinstance(csv_bytes, bytes)
    assert "Detector,Accuracy (%)" in csv_bytes.decode("utf-8")
    assert "ADWIN,90.0" in csv_bytes.decode("utf-8")


def test_obtener_metricas_ganador_extracts_key_numbers():
    tabla = pd.DataFrame(
        {
            "Detector": ["ADWIN", "DDM"],
            "Accuracy (%)": [94.2, 80.0],
            "F1 (%)": [93.0, 75.0],
            "Retardo_promedio": [12.0, None],
            "Tiempo_seg": [1.5, 2.0],
        }
    )
    metricas = obtener_metricas_ganador(tabla, "ADWIN")
    assert metricas == {
        "detector": "ADWIN",
        "accuracy": 94.2,
        "f1": 93.0,
        "retardo": 12.0,
        "tiempo": 1.5,
    }


def test_obtener_metricas_ganador_returns_none_without_winner():
    tabla = pd.DataFrame({"Detector": ["ADWIN"], "Accuracy (%)": [94.2]})
    assert obtener_metricas_ganador(tabla, None) is None


def test_resaltar_fila_ganadora_returns_a_styler_without_changing_data():
    tabla = pd.DataFrame({"Detector": ["ADWIN", "DDM"], "Accuracy (%)": [94.2, 80.0]})
    estilo = _resaltar_fila_ganadora(tabla, "ADWIN")
    assert isinstance(estilo, pd.io.formats.style.Styler)
    pd.testing.assert_frame_equal(estilo.data, tabla)


def test_ejecutar_todos_los_detectores_reports_progress_for_each_detector():
    X = np.random.default_rng(0).normal(size=(50, 2)).astype(np.float32)
    y = np.array([0, 1] * 25)
    llamadas = []
    _ejecutar_todos_los_detectores(
        X, y, tiene_labels=True, ground_truth=None, n_repeticiones=1,
        detector_params_por_nombre={}, progress_callback=lambda i, total, nombre: llamadas.append((i, total, nombre)),
    )
    assert len(llamadas) == 7
    assert llamadas[0][0] == 1 and llamadas[-1][0] == 7
    assert all(total == 7 for _, total, _ in llamadas)


def test_calcular_rango_zoom_uses_ground_truth_when_available():
    rango = _calcular_rango_zoom(drifts_idx=[], ground_truth=[5000], total_filas=14000, margen=500)
    assert rango == (4500, 5500)


def test_calcular_rango_zoom_falls_back_to_detected_drifts():
    rango = _calcular_rango_zoom(drifts_idx=[100, 900], ground_truth=None, total_filas=14000, margen=200)
    assert rango == (0, 1100)


def test_calcular_rango_zoom_returns_none_without_any_reference_point():
    assert _calcular_rango_zoom(drifts_idx=[], ground_truth=None, total_filas=14000) is None

import pandas as pd

from app import _graficar_metricas_comparativas, recomendar


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

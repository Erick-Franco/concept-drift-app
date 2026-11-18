import pandas as pd

from app import recomendar


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

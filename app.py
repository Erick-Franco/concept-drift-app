"""App Streamlit: comparacion de detectores de concept drift sobre un stream."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from detectores import DETECTOR_FACTORIES, run_detection
from io_utils import (
    count_rows,
    detect_label_column,
    encode_features,
    list_builtin_datasets,
    load_dataset,
    load_drift_ground_truth,
)

PRIORIDADES = ["Menor retardo", "Menos falsas alarmas", "Mayor Accuracy", "Menor costo computacional"]
MAX_MUESTRAS_DEFAULT = 5000


def recomendar(df_resultados: pd.DataFrame, prioridad: str) -> str:
    if df_resultados.empty:
        return "No hay resultados para recomendar."

    if prioridad == "Menor retardo":
        candidatos = df_resultados.dropna(subset=["Retardo_promedio"])
        if candidatos.empty:
            return "No hay ground truth disponible para evaluar el retardo en este dataset."
        fila = candidatos.loc[candidatos["Retardo_promedio"].idxmin()]
        return (
            f"**{fila['Detector']}** es el más recomendado: detecto el drift con un retardo "
            f"promedio de {fila['Retardo_promedio']:.1f} muestras, el menor de todos los algoritmos evaluados."
        )
    if prioridad == "Menos falsas alarmas":
        fila = df_resultados.loc[df_resultados["Drifts_Total"].idxmin()]
        return (
            f"**{fila['Detector']}** es el más recomendado: reporto solo {int(fila['Drifts_Total'])} "
            f"drifts en total, la menor cantidad (menos falsas alarmas potenciales)."
        )
    if prioridad == "Mayor Accuracy":
        fila = df_resultados.loc[df_resultados["Accuracy (%)"].idxmax()]
        return (
            f"**{fila['Detector']}** es el más recomendado: alcanzo {fila['Accuracy (%)']:.2f}% de "
            f"accuracy, la mas alta entre los algoritmos evaluados."
        )
    fila = df_resultados.loc[df_resultados["Tiempo_seg"].idxmin()]
    return (
        f"**{fila['Detector']}** es el más recomendado: tardo solo {fila['Tiempo_seg']:.2f}s en procesar "
        f"el stream, el menor costo computacional de todos."
    )


def _construir_tabla_resultados(dataset_name: str, resultados: list) -> pd.DataFrame:
    filas = []
    for r in resultados:
        filas.append(
            {
                "Dataset": dataset_name,
                "Detector": r.detector,
                "Accuracy (%)": r.accuracy * 100,
                "Drifts_Total": len(r.drifts_idx),
                "Primeros_Drifts": r.drifts_idx[:3],
                "Tiempo_seg": r.tiempo_seg,
                "RAM_Pico_KB": r.ram_pico_kb,
                "Retardo_promedio": r.retardo_promedio,
            }
        )
    return pd.DataFrame(filas)


def _graficar_stream(df: pd.DataFrame, label_col: str | None, resultados: list, ground_truth: list[int] | None):
    señal = df[label_col] if label_col and pd.api.types.is_numeric_dtype(df[label_col]) else df.iloc[:, 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=señal.to_numpy(), mode="lines", name="Señal del stream", line=dict(color="lightgray")))

    colores = ["red", "blue", "green", "orange", "purple", "brown", "pink"]
    for i, r in enumerate(resultados):
        for idx in r.drifts_idx:
            fig.add_vline(x=idx, line_color=colores[i % len(colores)], line_dash="dot", opacity=0.4)

    if ground_truth:
        for punto in ground_truth:
            fig.add_vline(x=punto, line_color="black", line_width=2)

    fig.update_layout(title="Stream y puntos de drift detectados", xaxis_title="Muestra", yaxis_title="Valor")
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="Detector de Concept Drift", layout="wide")
    st.title("Comparador de detectores de Concept Drift")

    builtin = list_builtin_datasets()
    st.sidebar.header("Configuración")
    opcion = st.sidebar.selectbox("Dataset", ["(subir CSV)"] + sorted(builtin.keys()))
    archivo_subido = st.sidebar.file_uploader("O sube tu propio CSV", type="csv")

    tiene_labels = st.sidebar.checkbox("¿El flujo tiene etiquetas inmediatas?", value=True)
    prioridad = st.sidebar.radio("Prioridad", PRIORIDADES)

    if archivo_subido is not None:
        total_filas = sum(1 for _ in archivo_subido) - 1
        archivo_subido.seek(0)
    elif opcion != "(subir CSV)":
        total_filas = count_rows(builtin[opcion])
    else:
        total_filas = MAX_MUESTRAS_DEFAULT

    slider_max = max(total_filas, 500)
    slider_default = total_filas if total_filas <= 15000 else MAX_MUESTRAS_DEFAULT
    max_muestras = st.sidebar.slider(
        "Máximo de muestras a procesar",
        min_value=500,
        max_value=slider_max,
        value=slider_default,
        step=500,
        help="El dataset se recorta a las primeras N muestras para mantener la app interactiva.",
    )
    ejecutar = st.sidebar.button("Ejecutar")

    if not ejecutar:
        st.info("Configura el dataset y las restricciones en la barra lateral, luego presiona Ejecutar.")
        return

    dataset_path: Path | None = None
    if archivo_subido is not None:
        df = pd.read_csv(archivo_subido)
        dataset_name = archivo_subido.name
        ground_truth = None
    elif opcion != "(subir CSV)":
        dataset_path = builtin[opcion]
        df = load_dataset(dataset_path)
        dataset_name = opcion
        ground_truth = load_drift_ground_truth(dataset_path)
    else:
        st.warning("Selecciona un dataset o sube un CSV.")
        return

    if len(df) > max_muestras:
        df = df.iloc[:max_muestras].reset_index(drop=True)

    label_col = detect_label_column(df)
    if tiene_labels and label_col is None:
        label_col = st.selectbox("No se detectó la columna de etiqueta automáticamente. Selecciónala:", df.columns)
    if not tiene_labels:
        label_col = None

    X, y, _ = encode_features(df, label_col if tiene_labels else None)

    resultados = []
    with st.spinner("Ejecutando detectores..."):
        for nombre in DETECTOR_FACTORIES:
            resultado = run_detection(X, y, nombre, tiene_labels=tiene_labels, ground_truth=ground_truth)
            resultados.append(resultado)

    tabla = _construir_tabla_resultados(dataset_name, resultados)
    st.subheader("Tabla comparativa")
    st.dataframe(tabla, use_container_width=True)

    st.subheader("Visualización del stream")
    _graficar_stream(df, label_col, resultados, ground_truth)

    st.subheader("Recomendación automática")
    st.success(recomendar(tabla, prioridad))


if __name__ == "__main__":
    main()

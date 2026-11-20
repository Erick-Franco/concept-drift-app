"""App Streamlit: comparacion de detectores de concept drift sobre un stream."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from detectores import (
    DETECTOR_DESCRIPTIONS,
    DETECTOR_FACTORIES,
    ejecutar_con_repeticiones,
)
from io_utils import (
    detect_label_column,
    encode_features,
    list_builtin_datasets,
    load_dataset,
    load_drift_ground_truth,
)

PRIORIDADES = ["Menor retardo", "Menos falsas alarmas", "Mayor Accuracy", "Menor costo computacional"]
MAX_MUESTRAS_DEFAULT = 5000


def elegir_ganador(df_resultados: pd.DataFrame, prioridad: str, tiene_labels: bool = True) -> str | None:
    """Devuelve el nombre del detector ganador segun la prioridad, o None
    cuando esa prioridad no se puede evaluar (sin datos, sin ground truth,
    o Accuracy sin etiquetas)."""
    if df_resultados.empty:
        return None
    if prioridad == "Mayor Accuracy" and not tiene_labels:
        return None
    if prioridad == "Menor retardo":
        candidatos = df_resultados.dropna(subset=["Retardo_promedio"])
        if candidatos.empty:
            return None
        return candidatos.loc[candidatos["Retardo_promedio"].idxmin(), "Detector"]
    if prioridad == "Menos falsas alarmas":
        return df_resultados.loc[df_resultados["Drifts_Total"].idxmin(), "Detector"]
    if prioridad == "Mayor Accuracy":
        return df_resultados.loc[df_resultados["Accuracy (%)"].idxmax(), "Detector"]
    return df_resultados.loc[df_resultados["Tiempo_seg"].idxmin(), "Detector"]


def contar_votos(ganadores: dict[str, str | None]) -> pd.Series:
    """Cuenta cuantas veces gano cada detector entre varios datasets,
    ignorando los datasets sin ganador (None)."""
    nombres = [g for g in ganadores.values() if g is not None]
    return pd.Series(nombres).value_counts()


def recomendar(df_resultados: pd.DataFrame, prioridad: str, tiene_labels: bool = True) -> str:
    if df_resultados.empty:
        return "No hay resultados para recomendar."

    if prioridad == "Mayor Accuracy" and not tiene_labels:
        return (
            "Sin etiquetas inmediatas no se puede calcular Accuracy (todos los detectores "
            "muestran 0%). Elige otra prioridad, por ejemplo 'Menor retardo' o 'Menos falsas alarmas'."
        )

    ganador = elegir_ganador(df_resultados, prioridad, tiene_labels)

    if prioridad == "Menor retardo":
        if ganador is None:
            return "No hay ground truth disponible para evaluar el retardo en este dataset."
        fila = df_resultados.loc[df_resultados["Detector"] == ganador].iloc[0]
        return (
            f"**{fila['Detector']}** es el más recomendado: detecto el drift con un retardo "
            f"promedio de {fila['Retardo_promedio']:.1f} muestras, el menor de todos los algoritmos evaluados."
        )
    fila = df_resultados.loc[df_resultados["Detector"] == ganador].iloc[0]
    if prioridad == "Menos falsas alarmas":
        return (
            f"**{fila['Detector']}** es el más recomendado: reporto solo {int(fila['Drifts_Total'])} "
            f"drifts en total, la menor cantidad (menos falsas alarmas potenciales)."
        )
    if prioridad == "Mayor Accuracy":
        return (
            f"**{fila['Detector']}** es el más recomendado: alcanzo {fila['Accuracy (%)']:.2f}% de "
            f"accuracy, la mas alta entre los algoritmos evaluados."
        )
    return (
        f"**{fila['Detector']}** es el más recomendado: tardo solo {fila['Tiempo_seg']:.2f}s en procesar "
        f"el stream, el menor costo computacional de todos."
    )


def obtener_metricas_ganador(tabla: pd.DataFrame, ganador: str | None) -> dict | None:
    """Extrae las metricas clave del detector ganador para mostrarlas como
    tarjetas (st.metric), en vez de que solo aparezcan dentro de la tabla."""
    if ganador is None:
        return None
    fila = tabla.loc[tabla["Detector"] == ganador].iloc[0]
    return {
        "detector": fila["Detector"],
        "accuracy": fila["Accuracy (%)"],
        "f1": fila["F1 (%)"],
        "retardo": fila["Retardo_promedio"],
        "tiempo": fila["Tiempo_seg"],
    }


def _resaltar_fila_ganadora(tabla: pd.DataFrame, ganador: str | None):
    """Devuelve un Styler que pinta la fila del detector ganador, para que
    resalte dentro de la tabla comparativa en vez de perderse entre las 7."""

    def _resaltar(fila):
        if ganador is not None and fila["Detector"] == ganador:
            return ["background-color: #1b4332; color: white"] * len(fila)
        return [""] * len(fila)

    return tabla.style.apply(_resaltar, axis=1)


def _calcular_rango_zoom(
    drifts_idx: list[int], ground_truth: list[int] | None, total_filas: int, margen: int = 500
) -> tuple[int, int] | None:
    """Calcula el rango del eje X para que el grafico del stream haga zoom
    automatico a la zona donde ocurre el drift, en vez de mostrar siempre
    el stream completo. Usa el ground truth si existe; si no, los puntos
    que detectaron los propios detectores. Sin ninguna referencia, no hay
    zoom (se muestra todo)."""
    puntos = list(ground_truth) if ground_truth else list(drifts_idx)
    if not puntos:
        return None
    inicio = max(0, min(puntos) - margen)
    fin = min(total_filas, max(puntos) + margen)
    return inicio, fin


def _construir_tabla_resultados(dataset_name: str, resultados: list) -> pd.DataFrame:
    filas = []
    for r in resultados:
        filas.append(
            {
                "Dataset": dataset_name,
                "Detector": r.detector,
                "Accuracy (%)": r.accuracy_mean * 100,
                "Accuracy_std (%)": r.accuracy_std * 100,
                "F1 (%)": r.f1_mean * 100,
                "F1_std (%)": r.f1_std * 100,
                "Drifts_Total": len(r.drifts_idx),
                "Primeros_Drifts": r.drifts_idx[:3],
                "Tiempo_seg": r.tiempo_seg_mean,
                "RAM_Pico_KB": r.ram_pico_kb_mean,
                "Retardo_promedio": r.retardo_promedio_mean,
            }
        )
    return pd.DataFrame(filas)


def _tabla_a_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _graficar_metricas_comparativas(df_resultados: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_resultados["Detector"], y=df_resultados["Accuracy (%)"], name="Accuracy (%)"))
    fig.add_trace(
        go.Bar(x=df_resultados["Detector"], y=df_resultados["Tiempo_seg"], name="Tiempo (s)", yaxis="y2")
    )
    fig.update_layout(
        title="Comparación de métricas por detector",
        barmode="group",
        yaxis=dict(title="Accuracy (%)"),
        yaxis2=dict(title="Tiempo (s)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def _graficar_stream(df: pd.DataFrame, label_col: str | None, resultados: list, ground_truth: list[int] | None):
    señal = df[label_col] if label_col and pd.api.types.is_numeric_dtype(df[label_col]) else df.iloc[:, 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=señal.to_numpy(), mode="lines", name="Señal del stream", line=dict(color="lightgray")))

    colores = ["red", "blue", "green", "orange", "purple", "brown", "pink"]
    todos_los_drifts: list[int] = []
    for i, r in enumerate(resultados):
        for idx in r.drifts_idx:
            fig.add_vline(x=idx, line_color=colores[i % len(colores)], line_dash="dot", opacity=0.4)
        todos_los_drifts.extend(r.drifts_idx)

    if ground_truth:
        for punto in ground_truth:
            fig.add_vline(x=punto, line_color="black", line_width=2)

    fig.update_layout(title="Stream y puntos de drift detectados", xaxis_title="Muestra", yaxis_title="Valor")

    rango_zoom = _calcular_rango_zoom(todos_los_drifts, ground_truth, len(df))
    if rango_zoom is not None:
        fig.update_xaxes(range=list(rango_zoom))
        st.caption("🔍 Gráfico con zoom automático a la zona del drift. Usa el ícono de Plotly para ver el stream completo.")

    st.plotly_chart(fig, use_container_width=True)


def _ejecutar_todos_los_detectores(
    X,
    y_para_deteccion,
    tiene_labels,
    ground_truth,
    n_repeticiones,
    detector_params_por_nombre,
    progress_callback=None,
) -> list:
    resultados = []
    total = len(DETECTOR_FACTORIES)
    for i, nombre in enumerate(DETECTOR_FACTORIES, start=1):
        resultado = ejecutar_con_repeticiones(
            X,
            y_para_deteccion,
            nombre,
            tiene_labels=tiene_labels,
            ground_truth=ground_truth,
            n_repeats=n_repeticiones,
            detector_params=detector_params_por_nombre.get(nombre),
        )
        resultados.append(resultado)
        if progress_callback is not None:
            progress_callback(i, total, nombre)
    return resultados


def _graficar_votos(votos: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(x=votos.index, y=votos.to_numpy()))
    fig.update_layout(
        title="Veces que cada detector ganó entre los datasets comparados",
        xaxis_title="Detector",
        yaxis_title="Datasets ganados",
    )
    return fig


def main():
    st.set_page_config(page_title="Detector de Concept Drift", layout="wide")
    st.title("Comparador de detectores de Concept Drift")

    with st.expander("Acerca de los detectores"):
        for nombre, descripcion in DETECTOR_DESCRIPTIONS.items():
            st.markdown(f"**{nombre}**: {descripcion}")

    st.sidebar.markdown(
        "<div style='text-align:center'>"
        "<span style='font-size:2rem'>🎓</span><br>"
        "<b>Universidad Peruana Unión</b><br>"
        "<small>Curso de Inteligencia Artificial</small>"
        "</div><hr>",
        unsafe_allow_html=True,
    )

    builtin = list_builtin_datasets()
    st.sidebar.header("Configuración")
    opcion = st.sidebar.selectbox("Dataset", ["(subir CSV)"] + sorted(builtin.keys()))
    archivo_subido = st.sidebar.file_uploader("O sube tu propio CSV", type="csv")

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
        st.info("Selecciona un dataset o sube un CSV, luego configura las restricciones en la barra lateral.")
        return

    tiene_labels = st.sidebar.checkbox("¿El flujo tiene etiquetas inmediatas?", value=True)
    prioridad = st.sidebar.radio("Prioridad", PRIORIDADES)

    # Se detecta (y, si hace falta, se pide) la columna de etiqueta AQUI, antes
    # del boton Ejecutar: un selectbox dibujado solo despues de hacer clic nunca
    # llega a usarse, porque ese mismo clic ya dispara la ejecucion sin esperar
    # a que el usuario lo cambie.
    label_col = detect_label_column(df)
    if tiene_labels and label_col is None:
        label_col = st.sidebar.selectbox(
            "No se detectó la columna de etiqueta automáticamente. Selecciónala:", df.columns
        )

    total_filas = len(df)
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
    with st.sidebar.expander("Parámetros avanzados"):
        adwin_delta = st.slider("ADWIN: delta (sensibilidad)", 0.0001, 0.1, 0.002, format="%.4f")
        page_threshold = st.slider("PageHinkley: threshold", 1, 100, 50)
        kswin_alpha = st.slider("KSWIN: alpha", 0.001, 0.2, 0.005, format="%.3f")
    detector_params_por_nombre = {
        "ADWIN": {"delta": adwin_delta},
        "PageHinkley": {"threshold": page_threshold},
        "KSWIN": {"alpha": kswin_alpha},
    }

    n_repeticiones = st.sidebar.slider(
        "Repeticiones (semillas)",
        min_value=1,
        max_value=5,
        value=1,
        help="Corre cada detector varias veces con distinta semilla y muestra "
        "media ± desviación estándar, más robusto que una sola corrida (más lento).",
    )
    ejecutar = st.sidebar.button("Ejecutar")

    st.sidebar.markdown("---")
    datasets_batch = st.sidebar.multiselect(
        "Comparar varios datasets (opcional)",
        sorted(builtin.keys()),
        help="Elige 2 o más datasets propios de la app para ver qué detector gana más veces.",
    )
    ejecutar_batch = st.sidebar.button("Ejecutar comparación batch")

    if ejecutar_batch and len(datasets_batch) >= 2:
        BATCH_MAX_MUESTRAS = 2000
        tablas_por_dataset = {}
        ganadores = {}
        with st.spinner("Comparando datasets..."):
            for nombre_ds in datasets_batch:
                path_ds = builtin[nombre_ds]
                df_ds = load_dataset(path_ds)
                gt_ds = load_drift_ground_truth(path_ds)
                label_col_ds = detect_label_column(df_ds)
                if tiene_labels and label_col_ds is None:
                    st.warning(f"{nombre_ds}: no se detectó columna de etiqueta, se omite.")
                    continue
                if len(df_ds) > BATCH_MAX_MUESTRAS:
                    df_ds = df_ds.iloc[:BATCH_MAX_MUESTRAS].reset_index(drop=True)
                X_ds, y_ds, _ = encode_features(df_ds, label_col_ds)
                y_para_ds = y_ds if tiene_labels else None
                try:
                    resultados_ds = _ejecutar_todos_los_detectores(
                        X_ds, y_para_ds, tiene_labels, gt_ds, n_repeticiones, detector_params_por_nombre
                    )
                except ValueError as e:
                    st.warning(f"{nombre_ds}: {e}")
                    continue
                tabla_ds = _construir_tabla_resultados(nombre_ds, resultados_ds)
                tablas_por_dataset[nombre_ds] = tabla_ds
                ganadores[nombre_ds] = elegir_ganador(tabla_ds, prioridad, tiene_labels)

        st.subheader("Comparación entre datasets")
        votos = contar_votos(ganadores)
        if votos.empty:
            st.info("Ningún dataset tuvo un ganador evaluable para esta prioridad.")
        else:
            st.plotly_chart(_graficar_votos(votos), use_container_width=True)
        for nombre_ds, tabla_ds in tablas_por_dataset.items():
            with st.expander(f"Detalle: {nombre_ds} (ganador: {ganadores[nombre_ds] or 'N/A'})"):
                st.dataframe(tabla_ds, use_container_width=True)
        return

    if not ejecutar:
        st.info("Configura el dataset y las restricciones en la barra lateral, luego presiona Ejecutar.")
        return

    if len(df) > max_muestras:
        df = df.iloc[:max_muestras].reset_index(drop=True)

    # label_col se excluye siempre de las features (haya o no etiquetas en uso):
    # de lo contrario, en modo sin etiquetas la columna de respuesta real se
    # filtraria como feature hacia el autoencoder.
    X, y, _ = encode_features(df, label_col)
    y_para_deteccion = y if tiene_labels else None

    barra_progreso = st.progress(0.0, text="Ejecutando detectores...")

    def _actualizar_progreso(i: int, total: int, nombre: str) -> None:
        barra_progreso.progress(i / total, text=f"Ejecutando {nombre}... ({i}/{total})")

    try:
        resultados = _ejecutar_todos_los_detectores(
            X,
            y_para_deteccion,
            tiene_labels,
            ground_truth,
            n_repeticiones,
            detector_params_por_nombre,
            progress_callback=_actualizar_progreso,
        )
    except ValueError as e:
        barra_progreso.empty()
        st.error(str(e))
        return
    barra_progreso.empty()

    tabla = _construir_tabla_resultados(dataset_name, resultados)
    ganador = elegir_ganador(tabla, prioridad, tiene_labels)
    metricas_ganador = obtener_metricas_ganador(tabla, ganador)

    st.subheader("Recomendación automática")
    if metricas_ganador is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏆 Detector recomendado", metricas_ganador["detector"])
        c2.metric("Accuracy", f"{metricas_ganador['accuracy']:.1f}%")
        c3.metric("F1-score", f"{metricas_ganador['f1']:.1f}%")
        retardo = metricas_ganador["retardo"]
        c4.metric("Retardo", f"{retardo:.0f} muestras" if retardo is not None else "N/A")
    st.success(recomendar(tabla, prioridad, tiene_labels))

    tab_tabla, tab_graficos = st.tabs(["📋 Resultados", "📊 Gráficos"])
    with tab_tabla:
        st.dataframe(_resaltar_fila_ganadora(tabla, ganador), use_container_width=True)
        st.download_button(
            "Descargar tabla (CSV)",
            data=_tabla_a_csv(tabla),
            file_name=f"comparacion_{dataset_name}.csv",
            mime="text/csv",
        )
    with tab_graficos:
        st.plotly_chart(_graficar_metricas_comparativas(tabla), use_container_width=True)
        _graficar_stream(df, label_col, resultados, ground_truth)


if __name__ == "__main__":
    main()

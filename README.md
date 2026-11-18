# Detector de Concept Drift — Proyecto final IA (UPeU)

Aplicación web interactiva (Streamlit) para subir o seleccionar un dataset, configurar
restricciones del flujo (¿tiene etiquetas inmediatas?, prioridad), ejecutar los
detectores de concept drift analizados en la RSL (ADWIN, DDM, EDDM, HDDM_A, HDDM_W,
PageHinkley, KSWIN) junto con un clasificador adaptativo (MLP) y un detector no
supervisado basado en autoencoder, y comparar resultados con métricas y gráficos.

## Instalación

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar la app

```bash
streamlit run app.py
```

## Ejecutar las pruebas

```bash
pytest
```

## Datasets

La app detecta automáticamente los datasets incluidos en:

- `data/data/` (NSL-KDD binario, sintético Hyperplane, cluster emergente)
- `datos_sinteticos/sinteticos/` (SEA/Hyperplane/LowRank, con ground truth de drift en `drifts.json`)

También se puede subir un CSV propio desde la barra lateral.

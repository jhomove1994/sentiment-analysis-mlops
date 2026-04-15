# Sentiment Analysis MLOps Pipeline

Pipeline completo de MLOps para análisis de sentimientos sobre tweets relacionados con Apple, desarrollado como parte del proyecto transversal de Innovación en Valoración Financiera con Aprendizaje Automático.

## 🎯 Objetivo

Diseñar e implementar un pipeline de CI/CD que facilite el entrenamiento, evaluación, despliegue y monitorización de un modelo de clasificación de sentimientos en un entorno de producción.

## 🏗️ Arquitectura

```
sentiment-analysis-mlops/
├── .github/
│   └── workflows/
│       └── ci_cd.yml          # Pipeline CI/CD con GitHub Actions
├── data/
│   └── sentiment_data.csv     # Dataset de tweets (Apple Twitter Sentiment)
├── src/
│   ├── train.py               # Entrenamiento del modelo con MLflow
│   ├── app.py                 # API REST con Flask + ngrok
│   └── monitor.py             # Monitorización: data drift + alertas de rendimiento
├── tests/
│   └── test_model.py          # Tests unitarios con pytest
├── metrics.json               # Métricas del último entrenamiento
├── requirements.txt           # Dependencias del proyecto
└── README.md
```

## 🔄 Pipeline CI/CD

El pipeline de GitHub Actions ejecuta automáticamente en cada `push` o `pull_request` a `main`:

| Paso | Descripción |
|------|-------------|
| 1. Instalar dependencias | `pip install -r requirements.txt` |
| 2. Generar datos de prueba | Crea dataset sintético de 300 tweets balanceados |
| 3. Entrenar modelo | `python src/train.py` — RandomForest + TF-IDF con MLflow |
| 4. Ejecutar tests | `pytest tests/ -v` — 4 tests unitarios |
| 5. Validar AUC | Verifica que AUC ≥ 0.60 |
| 6. Monitorizar | `python src/monitor.py` — detecta data drift y alertas |
| 7. Guardar artefacto | Modelo `.joblib` guardado como artefacto de GitHub Actions |

## 🚀 Ejecución local (Google Colab)

### Paso 1: Configuración del entorno
```python
# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Instalar dependencias
!pip install mlflow pyngrok flask scikit-learn evidently joblib pandas numpy pytest -q
```

### Paso 2: Clonar el repositorio
```bash
git clone https://github.com/jhomove1994/sentiment-analysis-mlops.git
cd sentiment-analysis-mlops
```

### Paso 3: Entrenar el modelo
```bash
python src/train.py
```

### Paso 4: Ejecutar tests
```bash
pytest tests/ -v
```

### Paso 5: Monitorizar el modelo
```bash
python src/monitor.py
```

### Paso 6: Lanzar la API
```bash
python src/app.py
```

## 🤖 Modelo

- **Algoritmo:** RandomForestClassifier (100 estimadores)
- **Vectorización:** TF-IDF (5000 features, unigramas y bigramas)
- **Clases:** `-1` (negativo), `0` (neutro), `1` (positivo)
- **Tracking:** MLflow (experimento `sentiment_analysis`)

## 📊 Métricas (último run)

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.8193 |
| AUC (OvR weighted) | 0.9198 |

## 🔍 Monitorización

El script `src/monitor.py` implementa:
- **Alerta de rendimiento:** verifica que Accuracy ≥ 0.70 y AUC ≥ 0.70
- **Detección de data drift:** test de Kolmogorov-Smirnov (umbral p < 0.05) comparando la distribución de los textos de entrada

## 🔧 Gestión de versiones

- **Código:** Git + GitHub
- **Modelo y métricas:** MLflow (tracking de experimentos, hiperparámetros y métricas)
- **Artefactos:** GitHub Actions artifacts + Google Drive (`/content/drive/MyDrive/mlops_sentiment/`)

## 📋 Dependencias

```
pandas
scikit-learn
mlflow
flask
pyngrok
evidently
joblib
numpy
pytest
```

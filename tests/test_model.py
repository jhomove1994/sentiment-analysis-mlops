import pytest, joblib, json, os


@pytest.fixture
def modelo():
    assert os.path.exists('modelo_sentimiento.joblib')
    return joblib.load('modelo_sentimiento.joblib')


def test_modelo_cargado(modelo):
    assert modelo is not None


def test_prediccion_valida(modelo):
    preds = modelo.predict(['Good results this quarter'])
    assert preds[0] in [0, 1]


def test_metricas_umbral():
    with open('metrics.json') as f: m = json.load(f)
    assert m['auc'] >= 0.70, f"AUC {m['auc']:.3f} bajo umbral"
    assert m['accuracy'] >= 0.70, f"Accuracy {m['accuracy']:.3f} bajo umbral"


def test_formato_proba(modelo):
    probs = modelo.predict_proba(['Test text'])
    assert 0 <= probs[0][1] <= 1
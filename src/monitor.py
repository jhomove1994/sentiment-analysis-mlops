import pandas as pd
import json
import os
from scipy.stats import ks_2samp


def check_data_drift(ref_path: str, new_path: str, threshold: float = 0.05) -> dict:
    """Detecta data drift comparando la distribución de longitudes de texto."""
    ref = pd.read_csv(ref_path)
    new = pd.read_csv(new_path)
    ref_lengths = ref['tweets'].str.len().values
    new_lengths = new['tweets'].str.len().values
    stat, p_value = ks_2samp(ref_lengths, new_lengths)
    drift_detected = p_value < threshold
    result = {
        'ks_statistic': round(float(stat), 4),
        'p_value': round(float(p_value), 4),
        'drift_detected': drift_detected,
        'threshold': threshold
    }
    status = 'SÍ ⚠️' if drift_detected else 'NO ✅'
    print(f"[Data Drift] KS stat={stat:.4f}, p={p_value:.4f} → drift detectado: {status}")
    return result


def check_performance_alert(metrics_path='metrics.json', auc_threshold=0.70, acc_threshold=0.70):
    """Lee metrics.json y lanza alertas si las métricas caen bajo los umbrales."""
    with open(metrics_path) as f:
        m = json.load(f)
    alerts = []
    if m['auc'] < auc_threshold:
        alerts.append(f"⚠️  AUC {m['auc']:.3f} por debajo del umbral {auc_threshold}")
    if m['accuracy'] < acc_threshold:
        alerts.append(f"⚠️  Accuracy {m['accuracy']:.3f} por debajo del umbral {acc_threshold}")
    if alerts:
        for alert in alerts: print(f"[Alerta] {alert}")
    else:
        print(f"[Rendimiento OK] AUC={m['auc']:.3f} | Accuracy={m['accuracy']:.3f}")
    return {'alerts': alerts, 'metrics': m}


def run_monitoring(ref_path='data/sentiment_data.csv', new_path=None, metrics_path='metrics.json'):
    """Ciclo completo de monitorización."""
    print("=" * 50)
    print("  MONITORIZACIÓN DEL MODELO DE SENTIMIENTOS")
    print("=" * 50)
    print("\n--- Paso 1: Verificación de métricas ---")
    perf = check_performance_alert(metrics_path=metrics_path)
    drift_result = None
    if new_path and os.path.exists(new_path):
        print("\n--- Paso 2: Detección de data drift ---")
        drift_result = check_data_drift(ref_path=ref_path, new_path=new_path)
    elif ref_path and os.path.exists(ref_path):
        print("\n--- Paso 2: Data drift (self-test) ---")
        drift_result = check_data_drift(ref_path=ref_path, new_path=ref_path)
    print("\n" + "=" * 50)
    total_alerts = len(perf.get('alerts', []))
    if total_alerts == 0 and (drift_result is None or not drift_result.get('drift_detected')):
        print("✅ Monitorización completada sin alertas.")
    else:
        print(f"⚠️  {total_alerts} alerta(s) de rendimiento.")
        if drift_result and drift_result.get('drift_detected'):
            print("⚠️  Data drift detectado — considerar reentrenamiento.")
    print("=" * 50)
    return {'performance': perf, 'drift': drift_result}


if __name__ == '__main__':
    run_monitoring()
import mlflow, mlflow.sklearn, joblib, json, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import os


# Configurar MLflow
if os.path.exists('/content'):
    DRIVE_BASE = '/content/drive/MyDrive/mlops_sentiment'
    mlflow.set_tracking_uri(f'file:///{DRIVE_BASE}/mlruns')
else:
    DRIVE_BASE = os.getcwd()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment('sentiment_analysis')


# Cargar datos
df = pd.read_csv('data/sentiment_data.csv')
X_train,X_test,y_train,y_test = train_test_split(
    df['tweets'], df['labels'], test_size=0.2, stratify=df['labels'], random_state=42)


with mlflow.start_run(run_name='rf_tfidf_v1') as run:
    N_EST=100; MAX_FEAT=5000
    mlflow.log_params({'n_estimators':N_EST,'max_features':MAX_FEAT})


    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=MAX_FEAT, ngram_range=(1,2))),
        ('clf',   RandomForestClassifier(n_estimators=N_EST, random_state=42))
    ])
    pipeline.fit(X_train, y_train)


    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    mlflow.log_metrics({'accuracy': acc, 'auc': auc})
    mlflow.sklearn.log_model(pipeline,'model',
        registered_model_name='sentiment_classifier')

    os.makedirs(f'{DRIVE_BASE}/models', exist_ok=True)
    joblib.dump(pipeline,'modelo_sentimiento.joblib')
    joblib.dump(pipeline,f'{DRIVE_BASE}/models/modelo_sentimiento.joblib')


    with open('metrics.json','w') as f:
        json.dump({'accuracy':acc,'auc':auc,'run_id':run.info.run_id},f)
    print(f'✓ Entrenamiento OK | Accuracy={acc:.4f} | AUC={auc:.4f}')
    print(f'  Run ID: {run.info.run_id}')
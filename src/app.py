from flask import Flask, request, jsonify
from pyngrok import ngrok
import joblib
from datetime import datetime


app = Flask(__name__)
model = joblib.load('modelo_sentimiento.joblib')
log = []


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('tweets','')
    pred = int(model.predict([text])[0])
    prob = float(model.predict_proba([text])[0][1])
    log.append({'ts':datetime.now().isoformat(),'pred':pred,'prob':prob})
    return jsonify({'prediction':pred,'probability':round(prob,3),
                    'sentiment':'positivo' if pred==1 else 'negativo'})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','n_predictions':len(log)})


if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f'API disponible en: {public_url}')
    app.run(host='0.0.0.0', port=5000)
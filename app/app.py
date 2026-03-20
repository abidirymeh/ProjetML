# app/app.py

from flask import Flask, request, render_template, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os
import sys
import csv
from datetime import datetime
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_model

app = Flask(__name__)

# Charger les modèles
print(" Démarrage de l'application...")
model = load_model('best_model.pkl', model_dir='../models')
pca = load_model('pca.pkl', model_dir='../models')

X_train_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'train_test', 'X_train.csv')
X_train_sample = pd.read_csv(X_train_path, nrows=1)
feature_names = X_train_sample.columns.tolist()

print(f" {len(feature_names)} features chargées")

if model is None or pca is None:
    print(" Modèles non trouvés")
else:
    print(" Modèles chargés avec succès")

os.makedirs('static', exist_ok=True)
reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
if os.path.exists(reports_dir):
    for img in ['confusion_matrix.png', 'roc_curve.png', 'pca_variance.png', 'clustering_analysis.png', 'feature_importance.png']:
        src = os.path.join(reports_dir, img)
        if os.path.exists(src):
            shutil.copy(src, f'static/{img}')
            print(f"Copié: {img}")

# Configuration historique
HISTORY_FILE = 'predictions_history.csv'

def create_full_features(recency, frequency, monetary, age=0, satisfaction=0, tickets=0):
    """Crée un DataFrame avec tous les features"""
    features = {name: 0 for name in feature_names}
    
    # Remplacer les valeurs
    for col in feature_names:
        if col == 'F2':
            features[col] = recency
        elif col == 'F3':
            features[col] = frequency
        elif col == 'F4':
            features[col] = monetary
        elif col == 'F31':  # Age
            features[col] = age
        elif col == 'F36':  # Satisfaction
            features[col] = satisfaction
        elif col == 'F35':  # Tickets support
            features[col] = tickets
    
    return pd.DataFrame([features])

def save_prediction(data, result):
    """Sauvegarde dans l'historique"""
    file_exists = os.path.isfile(HISTORY_FILE)
    with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Recency', 'Frequency', 'Monetary', 'Age', 'Satisfaction', 'Tickets', 'Prediction', 'Probabilite'])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data.get('recency', 0),
            data.get('frequency', 0),
            data.get('monetary', 0),
            data.get('age', 0),
            data.get('satisfaction', 0),
            data.get('tickets', 0),
            result['prediction'],
            result['probabilite']
        ])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer toutes les données
        recency = float(request.form.get('F2', 0))
        frequency = float(request.form.get('F3', 0))
        monetary = float(request.form.get('F4', 0))
        age = float(request.form.get('F31', 0))
        satisfaction = float(request.form.get('F36', 0))
        tickets = float(request.form.get('F35', 0))
        
        print(f" Données: Recency={recency}, Frequency={frequency}, Monetary={monetary}, Age={age}, Satisfaction={satisfaction}, Tickets={tickets}")
        
        # Créer DataFrame
        df = create_full_features(recency, frequency, monetary, age, satisfaction, tickets)
        
        # Appliquer PCA
        df_pca = pca.transform(df)
        
        # Prédire
        pred = model.predict(df_pca)[0]
        proba = model.predict_proba(df_pca)[0]
        
        result = {
            'prediction': int(pred),
            'probabilite': float(proba[1]),
            'statut': ' RISQUE DE CHURN' if pred == 1 else ' CLIENT FIDÈLE',
            'confiance': float(max(proba)),
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary,
            'age': age,
            'satisfaction': satisfaction,
            'tickets': tickets
        }
        
        # Sauvegarder dans l'historique
        save_prediction({
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary,
            'age': age,
            'satisfaction': satisfaction,
            'tickets': tickets
        }, result)
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        print(f" Erreur: {e}")
        return f"Erreur: {str(e)}"

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        recency = float(data.get('F2', 0))
        frequency = float(data.get('F3', 0))
        monetary = float(data.get('F4', 0))
        age = float(data.get('F31', 0))
        satisfaction = float(data.get('F36', 0))
        tickets = float(data.get('F35', 0))
        
        df = create_full_features(recency, frequency, monetary, age, satisfaction, tickets)
        df_pca = pca.transform(df)
        pred = model.predict(df_pca)[0]
        proba = model.predict_proba(df_pca)[0]
        
        return jsonify({
            'prediction': int(pred),
            'probabilite_churn': float(proba[1]),
            'message': 'Risque de churn' if pred == 1 else 'Client fidèle'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')
@app.route('/history')
def history():
    """Affiche l'historique des prédictions"""
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        
        history_data = df.to_dict('records')
        
        expected_columns = ['Date', 'Recency', 'Frequency', 'Monetary', 'Age', 'Satisfaction', 'Tickets', 'Prediction', 'Probabilite']
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        history_data = df.to_dict('records')
        
        stats = {
            'total': len(df),
            'churn_count': len(df[df['Prediction'] == 1]) if 'Prediction' in df.columns else 0,
            'avg_probabilite': df['Probabilite'].mean() * 100 if 'Probabilite' in df.columns else 0,
            'last_prediction': df.iloc[-1]['Date'] if len(df) > 0 else 'Aucune'
        }
        
        return render_template('history.html', history_data=history_data, stats=stats)
    
    return render_template('history.html', history_data=[], stats={'total': 0, 'churn_count': 0, 'avg_probabilite': 0, 'last_prediction': 'Aucune'})

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
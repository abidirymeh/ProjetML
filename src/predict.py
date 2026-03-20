# src/predict.py

import pandas as pd
import numpy as np
import joblib
import os
from utils import load_model

print("="*60)
print(" PRÉDICTION DU CHURN POUR NOUVEAUX CLIENTS")
print("="*60)

# 1. Charger les modèles
print("\n Chargement des modèles")
try:
    model = load_model('best_model.pkl')
    pca = load_model('pca.pkl')
    print(" Modèles chargés avec succès")
except FileNotFoundError:
    print(" Modèles non trouvés")
    exit(1)

# 2. Fonction de prédiction
def predict_single_client(client_data):
    
    # Convertir en DataFrame
    df = pd.DataFrame([client_data])
    
    # Appliquer PCA
    df_pca = pca.transform(df)
    
    # Prédire
    prediction = model.predict(df_pca)[0]
    proba = model.predict_proba(df_pca)[0]
    
    return prediction, proba

def predict_from_csv(csv_path):
    """Prédit pour plusieurs clients depuis un fichier CSV"""
    df = pd.read_csv(csv_path)
    df_pca = pca.transform(df)
    predictions = model.predict(df_pca)
    probas = model.predict_proba(df_pca)
    return predictions, probas

# 3. Exemple d'utilisation
print("\n EXEMPLE DE PRÉDICTION POUR UN CLIENT")
print("-" * 50)

# Créer un exemple de client 
exemple_client = {
    'F2': 30,   # Recency: 30 jours
    'F3': 10,   # Frequency: 10 commandes
    'F4': 500,  # MonetaryTotal: 500€
}

print("\nClient exemple:")
for key, value in exemple_client.items():
    print(f"  {key}: {value}")

# Prédire
try:
    pred, proba = predict_single_client(exemple_client)
    
    print(f"\n RÉSULTAT:")
    print(f"  Prédiction: {' RISQUE DE CHURN' if pred == 1 else ' CLIENT FIDÈLE'}")
    print(f"  Probabilité: {proba[1]:.2%} de churn")
    print(f"  Confiance: {max(proba):.2%}")
    
except Exception as e:
    print(f" Erreur: {e}")
    print("   Vérifiez que toutes les features nécessaires sont présentes")


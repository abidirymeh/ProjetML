# src/validation.py

import pandas as pd
import numpy as np
import os

print("\n" + "="*60)
print("VALIDATION DES PHASES PRE-MODELISATION")
print("="*60)

# Définir le chemin de base (racine du projet)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"\n📁 Répertoire de base: {BASE_DIR}")

# 1. Chargement des données nettoyées
processed_path = os.path.join(BASE_DIR, 'data', 'processed', 'retail_customers_clean.csv')

if not os.path.exists(processed_path):
    print(f"\n❌ ERREUR: Fichier {processed_path} non trouve!")
    print("Executez d'abord: python src/preprocessing.py")
    exit(1)

print(f"\n✅ Chargement des donnees depuis {processed_path}...")
data = pd.read_csv(processed_path)
print(f"✅ Dimensions: {data.shape}")

# 2. VERIFICATION DES FICHIERS train/test
print("\n🔍 VERIFICATION DES FICHIERS TRAIN/TEST:")
print("-" * 50)

train_files = {
    'X_train.csv': os.path.join(BASE_DIR, 'data', 'train_test', 'X_train.csv'),
    'X_test.csv': os.path.join(BASE_DIR, 'data', 'train_test', 'X_test.csv'),
    'y_train.csv': os.path.join(BASE_DIR, 'data', 'train_test', 'y_train.csv'),
    'y_test.csv': os.path.join(BASE_DIR, 'data', 'train_test', 'y_test.csv')
}

all_exist = True
for name, path in train_files.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024)  # Taille en KB
        print(f"✅ {name}: {size:.2f} KB")
    else:
        print(f"❌ {name}: manquant à {path}")
        all_exist = False

# 3. VERIFICATIONS des données
print("\n📊 VERIFICATIONS DES DONNEES:")
print("-" * 50)

# 3.1 Valeurs manquantes
missing = data.isnull().sum().sum()
if missing == 0:
    print(f"✅ Valeurs manquantes: 0")
else:
    print(f"❌ Valeurs manquantes: {missing} trouvées")

# 3.2 Types de données
non_numeric = data.select_dtypes(include=['object']).shape[1]
if non_numeric == 0:
    print(f"✅ Colonnes non-numeriques: 0")
else:
    print(f"❌ Colonnes non-numeriques: {non_numeric} trouvées")

# 3.3 Normalisation
feature_cols = [c for c in data.columns if c != 'Churn']
if len(feature_cols) > 0:
    sample_cols = feature_cols[:5]
    means = data[sample_cols].mean().abs().max()
    stds = data[sample_cols].std().std()
    
    if means < 0.1 and abs(stds - 1) < 0.1:
        print(f"✅ Normalisation correcte (moyennes ~0, ecarts-types ~1)")
    else:
        print(f"⚠️ Normalisation a verifier (moyenne max: {means:.4f}, ecart-type: {stds:.4f})")

# 3.4 Distribution de Churn
if 'Churn' in data.columns:
    churn_dist = data['Churn'].value_counts(normalize=True)
    print(f"\n📈 Distribution de Churn:")
    print(f"   - Classe 0 (fidele): {churn_dist.get(0, 0):.1%} ({int(churn_dist.get(0, 0)*len(data))} clients)")
    print(f"   - Classe 1 (parti): {churn_dist.get(1, 0):.1%} ({int(churn_dist.get(1, 0)*len(data))} clients)")

# 3.5 Aperçu des plages
print(f"\n📋 Apercu des plages de valeurs (5 premieres features):")
for col in sample_cols:
    print(f"   - {col}: min={data[col].min():.3f}, max={data[col].max():.3f}")

# 4. CHARGEMENT ET VERIFICATION DES SPLITS
if all_exist:
    print("\n🔄 VERIFICATION DES SPLITS:")
    print("-" * 50)
    
    X_train = pd.read_csv(train_files['X_train.csv'])
    X_test = pd.read_csv(train_files['X_test.csv'])
    y_train = pd.read_csv(train_files['y_train.csv']).squeeze()
    y_test = pd.read_csv(train_files['y_test.csv']).squeeze()
    
    print(f"✅ X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"✅ y_train: {len(y_train)}, y_test: {len(y_test)}")
    
    # Verifier la stratification
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    
    print(f"\n📊 Distribution des splits:")
    print(f"   - Train: 0={train_dist.get(0, 0):.1%}, 1={train_dist.get(1, 0):.1%}")
    print(f"   - Test:  0={test_dist.get(0, 0):.1%}, 1={test_dist.get(1, 0):.1%}")
    
    if abs(train_dist.get(0, 0) - test_dist.get(0, 0)) < 0.01:
        print("✅ Stratification correcte")
    else:
        print("⚠️ Difference de distribution detectee")

print("\n" + "="*60)

if missing == 0 and non_numeric == 0 and all_exist:
    print(" TOUT EST PARFAIT ")

else:
    print(" DES PROBLEMES SUBSISTENT !")
    print("\n Actions a mener:")
    if not all_exist:
        print("   - Les fichiers train/test sont manquants!")
        print("   - Exécutez: python src/preprocessing.py")
    if missing > 0:
        print("   - Revoir l'imputation des valeurs manquantes")
    if non_numeric > 0:
        print("   - Verifier l'encodage des variables categorielles")
print("="*60)

print("\n RAPPORT DE VALIDATION:")
print("-" * 50)
print(f"Total clients: {len(data)}")
print(f"Nombre de features: {len(feature_cols)}")
print(f"Colonnes totales: {len(data.columns)}")
print(f"Memoire utilisee: {data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

print(" PREPROCESSING DES DONNÉES")


# 1. Chargement des données
observations = pd.read_csv('../data/raw/retail_customers_COMPLETE_CATEGORICAL.csv')
print(f"\n Dimensions initiales: {observations.shape}")

# 2. Renommage des colonnes
new_column_names = {}
feature_counter = 1
for col in observations.columns:
    if col == 'Churn':
        new_column_names[col] = 'Churn'
    else:
        new_column_names[col] = f'F{feature_counter}'
        feature_counter += 1
observations.rename(columns=new_column_names, inplace=True)
print(" Colonnes renommées")

# 3. Distribution de la cible
print("\n Distribution Churn:")
print(observations.groupby('Churn').size())

# 4. Identification des types
numerical_cols = observations.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = observations.select_dtypes(include=['object']).columns
print(f"\n {len(numerical_cols)} colonnes numériques, {len(categorical_cols)} catégorielles")

# 5. Traitement des valeurs aberrantes - SupportTickets (F35)
print("\n--- TRAITEMENT F35 (SupportTickets) ---")
print(f"Avant: valeurs uniques = {observations['F35'].nunique()}")
observations.loc[observations['F35'] == -1, 'F35'] = np.nan
median_tickets = observations[observations['F35'] < 100]['F35'].median()
observations.loc[observations['F35'] > 50, 'F35'] = median_tickets
print(f"Après: valeurs uniques = {observations['F35'].nunique()}")

# 6. Traitement SatisfactionScore (F36)
print("\n--- TRAITEMENT F36 (Satisfaction) ---")
print(f"Avant: valeurs uniques = {observations['F36'].nunique()}")
observations.loc[observations['F36'] == -1, 'F36'] = np.nan
median_satisfaction = observations[observations['F36'] <= 5]['F36'].median()
observations.loc[observations['F36'] > 5, 'F36'] = median_satisfaction
print(f"Après: valeurs uniques = {observations['F36'].nunique()}")

# 7. Imputation Age (F31)
print("\n--- TRAITEMENT F31 (Age) ---")
age_median = observations['F31'].median()
observations['F31'].fillna(age_median, inplace=True)
print(f" Age: {observations['F31'].isnull().sum()} manquants après imputation")

# 8. Imputation autres colonnes
for col in ['F19', 'F35', 'F36']:
    median_val = observations[col].median()
    observations[col].fillna(median_val, inplace=True)

# 9. Encodage
print("\n--- ENCODAGE DES VARIABLES CATÉGORIELLES ---")
observations_encoded = observations.copy()

# Label Encoding pour variables ordinales
ordinal_cols = ['F37', 'F38', 'F40', 'F43', 'F44', 'F46']
for col in ordinal_cols:
    if col in categorical_cols:
        le = LabelEncoder()
        observations_encoded[col] = le.fit_transform(observations_encoded[col].astype(str))
        print(f" Encodé: {col}")

# One-Hot Encoding
nominal_cols = [col for col in categorical_cols if col not in ordinal_cols]
observations_encoded = pd.get_dummies(observations_encoded, columns=nominal_cols, drop_first=True)
print(f" One-Hot encoding: {len(nominal_cols)} colonnes → {observations_encoded.shape[1] - observations.shape[1]} nouvelles")

# 10. Normalisation
print("\n--- NORMALISATION ---")
feature_cols = [col for col in observations_encoded.columns if col != 'Churn']
scaler = StandardScaler()
observations_encoded[feature_cols] = scaler.fit_transform(observations_encoded[feature_cols])
print(f" {len(feature_cols)} features normalisées")

# 11. Split train/test
print("\n--- SPLIT TRAIN/TEST ---")
X = observations_encoded.drop('Churn', axis=1)
y = observations_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f" X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f" Distribution préservée: {y_train.value_counts(normalize=True).values}")

# 12. Sauvegarde
print("\n--- SAUVEGARDE ---")
os.makedirs('../data/train_test', exist_ok=True)
os.makedirs('../data/processed', exist_ok=True)

X_train.to_csv('../data/train_test/X_train.csv', index=False)
X_test.to_csv('../data/train_test/X_test.csv', index=False)
y_train.to_csv('../data/train_test/y_train.csv', index=False)
y_test.to_csv('../data/train_test/y_test.csv', index=False)
observations_encoded.to_csv('../data/processed/retail_customers_clean.csv', index=False)

print(" Données sauvegardées dans:")
print("   - data/train_test/ (splits)")
print("   - data/processed/retail_customers_clean.csv")
print("\n" + "="*60)
print("PREPROCESSING TERMINÉ AVEC SUCCÈS ! ")
print("="*60)
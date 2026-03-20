# src/utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import learning_curve
import joblib
import os

def load_data(data_path):
   
    try:
        df = pd.read_csv(data_path)
        print(f" Données chargées: {df.shape}")
        return df
    except Exception as e:
        print(f" Erreur chargement: {e}")
        return None

def save_model(model, filename, model_dir='../models'):
  
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)
    joblib.dump(model, path)
    print(f" Modèle sauvegardé: {path}")
    return path

def load_model(filename, model_dir='../models'):
   
    path = os.path.join(model_dir, filename)
    try:
        model = joblib.load(path)
        print(f" Modèle chargé: {path}")
        return model
    except Exception as e:
        print(f" Erreur chargement: {e}")
        return None

def plot_confusion_matrix(y_true, y_pred, labels=['Fidèle', 'Churn'], save_path=None):
   
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Matrice de confusion')
    plt.ylabel('Vrai')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f" Matrice sauvegardée: {save_path}")
    plt.show()
    plt.close()

def plot_roc_curve(y_true, y_proba, save_path=None):
   
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f" ROC sauvegardée: {save_path}")
    plt.show()
    plt.close()
    
    return roc_auc

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
  
    if not hasattr(model, 'feature_importances_'):
        print(" Le modèle n'a pas d'attribut feature_importances_")
        return
    
    importances = pd.DataFrame({
        'feature': feature_names[:len(model.feature_importances_)],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importances['feature'], importances['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features les plus importantes')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f" Importance sauvegardée: {save_path}")
    plt.show()
    plt.close()
    
    return importances

def plot_learning_curve(model, X, y, cv=5, save_path=None):
   
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Score entraînement')
    plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Score validation')
    plt.xlabel('Taille de l\'entraînement')
    plt.ylabel('Score')
    plt.title('Courbe d\'apprentissage')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f" Courbe sauvegardée: {save_path}")
    plt.show()
    plt.close()

def parse_registration_date(df, date_column):
   
    
    df[date_column + '_parsed'] = pd.to_datetime(df[date_column], 
                                                   dayfirst=True, 
                                                   errors='coerce')
    
    # Extraire les composantes
    df['RegYear'] = df[date_column + '_parsed'].dt.year
    df['RegMonth'] = df[date_column + '_parsed'].dt.month
    df['RegDay'] = df[date_column + '_parsed'].dt.day
    df['RegWeekday'] = df[date_column + '_parsed'].dt.weekday
    
    # Statistiques sur les dates manquantes
    missing = df[date_column + '_parsed'].isnull().sum()
    if missing > 0:
        print(f" {missing} dates non parsées")
    
    # Supprimer la colonne temporaire
    df.drop(date_column + '_parsed', axis=1, inplace=True)
    df.drop(date_column, axis=1, inplace=True)
    
    return df

def create_features(df):
  
    # Mapping des colonnes disponibles
    features = {}
    
    # Ratios métier pertinents
    if 'F4' in df.columns and 'F2' in df.columns:
        df['MonetaryPerDay'] = df['F4'] / (df['F2'] + 1)
        print(" Créé: MonetaryPerDay")
    
    if 'F4' in df.columns and 'F3' in df.columns:
        df['AvgBasketValue'] = df['F4'] / df['F3'].replace(0, 1)
        print(" Créé: AvgBasketValue")
    
    if 'F2' in df.columns and 'F13' in df.columns:
        df['TenureRatio'] = df['F2'] / (df['F13'] + 1)
        print(" Créé: TenureRatio")
    
    # Score RFM combiné
    if all(col in df.columns for col in ['F2', 'F3', 'F4']):
        # Normaliser chaque composante
        r_score = pd.qcut(df['F2'].rank(method='first'), 4, labels=False)  # Plus petit = meilleur
        f_score = pd.qcut(df['F3'].rank(method='first'), 4, labels=False)  # Plus grand = meilleur
        m_score = pd.qcut(df['F4'].rank(method='first'), 4, labels=False)  # Plus grand = meilleur
        
        df['RFM_Score'] = r_score + f_score + m_score
        print(" Créé: RFM_Score")
    
    return df

def detect_outliers(df, column, threshold=3):
   
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = z_scores > threshold
    print(f" {column}: {outliers.sum()} outliers détectés (seuil={threshold})")
    return outliers

def print_model_results(y_test, y_pred, y_proba=None):
  
    print("\n" + "="*50)
    print(" RÉSULTATS DU MODÈLE")
    print("="*50)
    
    # Accuracy
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Accuracy: {acc:.2%}")
    
    # Classification report
    print("\n Rapport de classification:")
    print(classification_report(y_test, y_pred, target_names=['Fidèle', 'Churn']))
    
    if y_proba is not None:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, y_proba)
        print(f"\n AUC-ROC: {auc:.3f}")

def ensure_dir(directory):
  
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f" Dossier créé: {directory}")
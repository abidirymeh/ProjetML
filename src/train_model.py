# src/train_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
import joblib
import os

# configuration pour les graphiques
plt.style.use('ggplot')
sns.set_palette("husl")

print("="*70)
print("MODELISATION COMPLETE - PREDICTION DU CHURN")
print("="*70)

# 1 chargemen t des données

print("\n" + "-"*70)
print("1. CHARGEMENT DES DONNEES")
print("-"*70)

try:
    X_train = pd.read_csv('../data/train_test/X_train.csv')
    X_test = pd.read_csv('../data/train_test/X_test.csv')
    y_train = pd.read_csv('../data/train_test/y_train.csv').squeeze()
    y_test = pd.read_csv('../data/train_test/y_test.csv').squeeze()
    
    print(f" X_train: {X_train.shape}")
    print(f" X_test: {X_test.shape}")
    print(f" y_train: {len(y_train)}")
    print(f" y_test: {len(y_test)}")
except FileNotFoundError as e:
    print(f" Erreur: {e}")
    print(" Executez d'abord: python src/preprocessing.py")
    exit(1)

# 2. TRANSFORMATION - ANALYSE EN COMPOSANTES PRINCIPALES (ACP)

print("\n" + "-"*70)
print("2. REDUCTION DE DIMENSION (ACP)")
print("-"*70)

# Standardisation pour vérifier
print("\nApplication de l'ACP...")

# Calculer le nombre de composantes 
pca_full = PCA()
pca_full.fit(X_train)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum >= 0.95) + 1

print(f"Nombre de composantes pour 95% de variance: {n_components_95}")

# ACP avec 95% de variance
pca = PCA(n_components=n_components_95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f" Dimensions avant ACP: {X_train.shape}")
print(f" Dimensions apres ACP: {X_train_pca.shape}")
print(f" Variance expliquee: {pca.explained_variance_ratio_.sum():.2%}")

# Visualisation de la variance expliquée
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} composantes')
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance expliquee cumulee')
plt.title('Selection du nombre de composantes ACP')
plt.legend()
plt.grid(True)
plt.savefig('../reports/pca_variance.png')
print(" Graphique sauvegarde: reports/pca_variance.png")

# 3. CLUSTERING (SEGMENTATION CLIENTS)

print("\n" + "-"*70)
print("3. CLUSTERING - SEGMENTATION CLIENTS")
print("-"*70)

# Tester différents nombres de clusters
silhouette_scores = []
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_train_pca[:, :10])  # Utiliser les 10 premieres PC
    
    silhouette = silhouette_score(X_train_pca[:, :10], labels)
    silhouette_scores.append(silhouette)
    inertias.append(kmeans.inertia_)
    
    print(f"K={k}: Silhouette={silhouette:.4f}, Inertie={kmeans.inertia_:.2f}")

# Meilleur K
best_k = K_range[np.argmax(silhouette_scores)]
print(f"\n Meilleur nombre de clusters: {best_k} (silhouette max)")

# Visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, silhouette_scores, 'bo-')
ax1.set_xlabel('Nombre de clusters (k)')
ax1.set_ylabel('Score de silhouette')
ax1.set_title('Selection du nombre de clusters')
ax1.grid(True)

ax2.plot(K_range, inertias, 'ro-')
ax2.set_xlabel('Nombre de clusters (k)')
ax2.set_ylabel('Inertie')
ax2.set_title('Methode du coude')
ax2.grid(True)

plt.tight_layout()
plt.savefig('../reports/clustering_analysis.png')
print(" Graphique sauvegarde: reports/clustering_analysis.png")

# Clustering final avec le meilleur K
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters_train = kmeans_final.fit_predict(X_train_pca[:, :10])
clusters_test = kmeans_final.predict(X_test_pca[:, :10])

# Analyser les clusters par rapport au churn
cluster_analysis = pd.DataFrame({
    'Cluster': clusters_train,
    'Churn': y_train.values
})
cluster_churn_rate = cluster_analysis.groupby('Cluster')['Churn'].agg(['mean', 'count'])
cluster_churn_rate.columns = ['Taux de churn', 'Nombre clients']
print("\nAnalyse des clusters:")
print(cluster_churn_rate)

# 4 CLASSIFICATION - MODELES DE BASE

print("\n" + "-"*70)
print("4. CLASSIFICATION - MODELES DE BASE")
print("-"*70)

models = {
    'KNN': KNeighborsClassifier(),
    'Arbre de decision': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}
predictions = {}
cv_results = {}

print("\n" + "-"*50)
print(f"{'Modele':20} {'Accuracy Test':15} {'CV Moyenne':15} {'CV Std':10}")
print("-"*50)

for name, model in models.items():
    # Validation croisée
    cv_scores = cross_val_score(model, X_train_pca, y_train, cv=5)
    cv_results[name] = {'mean': cv_scores.mean(), 'std': cv_scores.std()}
    
    # Entraînement et test
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    predictions[name] = y_pred
    
    print(f"{name:20} {acc:.4f}           {cv_scores.mean():.4f}        {cv_scores.std():.4f}")

# Meilleur modèle
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n Meilleur modele de base: {best_model_name} avec accuracy {results[best_model_name]:.4f}")


# 5. OPTIMISATION DES HYPERPARAMETRES 

print("\n" + "-"*70)
print("5. OPTIMISATION DES HYPERPARAMETRES (GridSearchCV) - VERSION RAPIDE")
print("-"*70)

# Optimisation pour Random Forest 
print("\nOptimisation de Random Forest (rapide)...")

param_grid_rf = {
    'n_estimators': [50, 100],           
    'max_depth': [10, None],             
    'min_samples_split': [2, 5],          
    'min_samples_leaf': [1, 2]            
}


print(f"🔍 Test de {len(param_grid_rf['n_estimators']) * len(param_grid_rf['max_depth']) * len(param_grid_rf['min_samples_split']) * len(param_grid_rf['min_samples_leaf'])} combinaisons")

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid_rf,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_rf.fit(X_train_pca, y_train)

print(f"\n Meilleurs parametres Random Forest:")
for param, value in grid_rf.best_params_.items():
    print(f"   - {param}: {value}")
print(f" Meilleure score CV: {grid_rf.best_score_:.4f}")

# Optimisation pour Gradient Boosting 
print("\nOptimisation de Gradient Boosting (rapide)...")

param_grid_gb = {
    'n_estimators': [50, 100],          
    'max_depth': [3, 5],                  
    'learning_rate': [0.1, 0.2],          
    'subsample': [0.8, 1.0]               
}

print(f" Test de {len(param_grid_gb['n_estimators']) * len(param_grid_gb['max_depth']) * len(param_grid_gb['learning_rate']) * len(param_grid_gb['subsample'])} combinaisons")

grid_gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid_gb,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_gb.fit(X_train_pca, y_train)

print(f"\n Meilleurs parametres Gradient Boosting:")
for param, value in grid_gb.best_params_.items():
    print(f"   - {param}: {value}")
print(f" Meilleure score CV: {grid_gb.best_score_:.4f}")
# 6 EVALUATION FINALE

print("\n" + "-"*70)
print("6. EVALUATION FINALE DES MODELES")
print("-"*70)

# Sélectionner le meilleur modèle optimisé
best_cv_score = max(grid_rf.best_score_, grid_gb.best_score_)
if grid_rf.best_score_ >= grid_gb.best_score_:
    final_model = grid_rf.best_estimator_
    print("\n Modele final selectionne: Random Forest optimise")
else:
    final_model = grid_gb.best_estimator_
    print("\n Modele final selectionne: Gradient Boosting optimise")

# Prédictions finales
y_pred_final = final_model.predict(X_test_pca)
y_pred_proba = final_model.predict_proba(X_test_pca)[:, 1]

# Métriques
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"\nAccuracy finale: {final_accuracy:.4f}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fidele', 'Churn'],
            yticklabels=['Fidele', 'Churn'])
plt.title(f'Matrice de confusion - {type(final_model).__name__}')
plt.ylabel('Vrai')
plt.xlabel('Prediction')
plt.tight_layout()
plt.savefig('../reports/confusion_matrix.png')
print(" Matrice de confusion sauvegardee: reports/confusion_matrix.png")

# Rapport de classification
print("\n" + "-"*50)
print("RAPPORT DE CLASSIFICATION")
print("-"*50)
print(classification_report(y_test, y_pred_final, target_names=['Fidele', 'Churn']))

# Courbe ROC-AUC
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('../reports/roc_curve.png')
print(" Courbe ROC sauvegardee: reports/roc_curve.png")

# Feature importance 
if hasattr(final_model, 'feature_importances_'):
    importances = pd.DataFrame({
        'composante': [f'PC{i+1}' for i in range(pca.n_components_)],
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importances['composante'].head(15), importances['importance'].head(15))
    plt.xlabel('Importance')
    plt.title('Top 15 composantes principales - Importance')
    plt.tight_layout()
    plt.savefig('../reports/feature_importance.png')
    print(" Importance des features sauvegardee: reports/feature_importance.png")

# 7 SAUVEGARDE DES MODELES

print("\n" + "-"*70)
print("7. SAUVEGARDE DES MODELES")
print("-"*70)

os.makedirs('../models', exist_ok=True)

# Sauvegarder le modele final
joblib.dump(final_model, '../models/best_model.pkl')
print(" Modele final sauvegarde: models/best_model.pkl")

# Sauvegarder le PCA
joblib.dump(pca, '../models/pca.pkl')
print(" PCA sauvegarde: models/pca.pkl")

# Sauvegarder le KMeans
joblib.dump(kmeans_final, '../models/kmeans.pkl')
print(" Modele de clustering sauvegarde: models/kmeans.pkl")


print("\n" + "-"*70)
print("8. RECOMMANDATIONS METIER")
print("-"*70)

print("\n ANALYSE DES RESULTATS:")
print("-"*50)

# Interpretation des clusters
print("\n SEGMENTATION CLIENTELE:")
print(f"   - {best_k} segments de clients identifies")
for cluster in range(best_k):
    rate = cluster_churn_rate.loc[cluster, 'Taux de churn']
    count = cluster_churn_rate.loc[cluster, 'Nombre clients']
    risk = "ELEVE" if rate > 0.5 else "MOYEN" if rate > 0.3 else "FAIBLE"
    print(f"   - Cluster {cluster}: {count} clients, taux de churn={rate:.1%} (RISQUE {risk})")

# Performance du modele
print("\n PERFORMANCE DU MODELE:")
print(f"   - Accuracy: {final_accuracy:.2%}")
print(f"   - AUC-ROC: {roc_auc:.2f}")
print(f"   - Modele utilise: {type(final_model).__name__}")

# 
print("\n RECOMMANDATIONS POUR L'ENTREPRISE:")
print("-"*50)
print("   1. Clients a risque eleve (>50%):")
print("      - Offrir des reductions personnalisees")
print("      - Envoyer des emails de fidelisation")
print("      - Proposer un programme de parrainage")
print("\n   2. Clients a risque moyen (30-50%):")
print("      - Envoyer des newsletters ciblees")
print("      - Proposer des produits complementaires")
print("      - Enquete de satisfaction")
print("\n   3. Clients fideles (risque faible):")
print("      - Programme VIP")
print("      - Offres exclusives")
print("      - Incitations au parrainage")

print("\n" + "="*70)
print(" MODELISATION COMPLETE TERMINEE AVEC SUCCES !")
print("="*70)
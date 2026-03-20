# Projet ML - Prédiction du Churn Client

##  Description
Ce projet consiste à développer une application de Machine Learning permettant de prédire le churn (désabonnement) des clients dans un e-commerce de cadeaux.

L'objectif est d'analyser le comportement des clients afin d’identifier ceux qui risquent de quitter la plateforme, et ainsi aider à la prise de décision marketing.

---

##  Installation

### 1. Créer un environnement virtuel
```bash
python -m venv venv
```

### 2. Activer l’environnement virtuel

- **Windows**
```bash
venv\Scripts\activate
```

- **Linux / Mac**
```bash
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

##  Structure du projet

```text
projet_ml_retail/
├── data/
│   ├── raw/                 # Données brutes
│   ├── processed/           # Données nettoyées
│   └── train_test/          # Données séparées train/test
├── notebooks/
│   └── 01_exploration.ipynb # Analyse exploratoire
├── src/
│   ├── preprocessing.py     # Prétraitement des données
│   ├── train_model.py       # Entraînement des modèles
│   ├── predict.py           # Prédictions
│   ├── validation.py        # Validation des données
│   └── utils.py             # Fonctions utilitaires
├── models/                  # Modèles sauvegardés
├── app/
│   ├── app.py               # Application Flask
│   └── templates/           # Interface HTML
├── reports/                 # Visualisations et graphiques
├── requirements.txt         # Dépendances Python
└── README.md                # Documentation du projet
```

---

##  Utilisation

### 1. Prétraitement des données
```bash
python src/preprocessing.py
```
- Nettoyage des données  
- Gestion des valeurs manquantes  
- Encodage des variables catégorielles  
- Normalisation des features  

---

### 2. Entraînement du modèle
```bash
python src/train_model.py
```
- Réduction de dimension (ACP)  
- Clustering  
- Classification  
- Optimisation des hyperparamètres  

---

### 3. Validation des données
```bash
python src/validation.py
```
- Vérification de la qualité des données  
- Validation du pipeline  

---

### 4. Lancer l’application web
```bash
cd app
python app.py
```

Puis ouvrir :
```
http://127.0.0.1:5000
```

---

### 5. Faire une prédiction
- Remplir le formulaire avec les données client  
- Cliquer sur **"Prédire"**  
- Visualiser le résultat  

---

##  Résultats

- **Accuracy** : 90.06%  
- **AUC-ROC** : 0.99  
- **Meilleur modèle** : Gradient Boosting  

---

##  Technologies utilisées

- Python  
- Pandas / NumPy  
- Scikit-learn  
- Flask  
- Matplotlib / Seaborn  

---



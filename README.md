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


Langage et environnement
Python 3.10
Jupyter Notebook (ipykernel 7.2.0, ipython 8.38.0)
VS Code 1.113.0

Traitement des données
NumPy 2.2.6 (calculs numériques et vectoriels)
Pandas 2.3.3 (manipulation et analyse de données tabulaires)
SciPy 1.15.3 (calculs scientifiques)

Machine Learning
Scikit-learn 1.7.2 (classification, ACP, clustering, GridSearchCV)
Joblib 1.5.3 (sauvegarde et chargement des modèles)
Threadpoolctl 3.6.0 (gestion des threads pour les calculs parallèles)

Visualisation
Matplotlib 3.10.8 (création de graphiques)
Seaborn 0.13.2 (visualisations statistiques avancées)
Pillow 12.1.1 (traitement d'images)

Déploiement web
Flask 3.1.3 (framework web pour l'application)
Werkzeug 3.1.6 (serveur WSGI)
Jinja2 3.1.6 (moteur de templates HTML)

Utilitaires
Tabulate 0.9.0 (formatage de tableaux)
Colorama 0.4.6 (couleurs dans le terminal)
Pygments 2.19.2 (coloration syntaxique)

---



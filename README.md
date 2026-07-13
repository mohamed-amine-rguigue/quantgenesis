# QuantGenesis

QuantGenesis est un projet de recherche et de démonstration pour l’analyse d’actions du secteur des semi-conducteurs à l’aide de techniques d’apprentissage automatique, de RAG et d’IA générative.

## Fonctionnalités

- Collecte de données de marché et d’actualités
- Calcul de features techniques
- Prédiction de tendance avec PyTorch et MLflow
- Analyse enrichie avec RAG, FAISS et un modèle LLM via Ollama
- Interface interactive avec Streamlit

## Architecture

- Data pipeline : collecte des prix et des actualités, stockage SQLite
- Machine learning : entraînement et inférence de modèle de classification
- RAG : embeddings, index FAISS, récupération de documents pertinents
- Interface : application Streamlit

## Prérequis

- Python 3.10+
- Ollama installé et lancé localement
- Dépendances Python listées dans requirements.txt

## Installation

```bash
cd quantgenesis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app/streamlit_app.py
```

## Environnement

Un fichier .env.example est fourni. Copiez-le vers .env et adaptez les valeurs si nécessaire.

## Docker

```bash
docker compose up --build
```

## Licence

Ce projet est distribué sous licence MIT.

# QuantGenesis

QuantGenesis est un projet de recherche et de démonstration pour l’analyse d’actions du secteur des semi-conducteurs à l’aide de techniques d’apprentissage automatique, de RAG et d’IA générative.

## Fonctionnalités

- Collecte de données de marché et d’actualités
- Calcul de features techniques
- Prédiction de tendance avec PyTorch et MLflow
- Analyse enrichie avec LangChain, RAG, FAISS et un modèle LLM via Ollama
- Intégration de modèles issus de l’écosystème Hugging Face
- Stockage et interrogation de données via SQL / SQLite
- Interface interactive avec Streamlit

## Architecture

- Data pipeline : collecte des prix et des actualités, stockage SQL / SQLite
- Machine learning : entraînement et inférence de modèle de classification avec PyTorch
- RAG : embeddings, index FAISS, récupération de documents pertinents
- LLM orchestration : LangChain et Ollama
- MLOps : suivi d’expériences avec MLflow
- Déploiement : conteneurisation avec Docker
- Interface : application Streamlit

## Méthodologie d’évaluation

Le modèle doit être évalué de manière temporelle, et non avec un split aléatoire, car il s’agit de séries financières. En pratique, il est préférable de :

- entraîner sur une période passée et valider sur une période suivante,
- répéter l’évaluation sur plusieurs fenêtres temporelles,
- comparer les performances à une baseline simple,
- vérifier que les features et les actualités utilisées ne contiennent pas d’information future.

Cette approche est plus fidèle à un usage réel et permet d’évaluer si le modèle généralise sur différents contextes de marché.

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

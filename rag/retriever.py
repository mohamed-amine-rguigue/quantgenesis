import os
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rag.embedder import embed_query, load_faiss_index, embed_articles, FAISS_INDEX_PATH

POSITIVE_KEYWORDS = ["record", "beat", "growth", "profit", "surge", "rally", "strong",
    "upgrade", "bullish", "outperform", "gain", "rise", "revenue", "partnership",
    "launch", "innovation", "buy", "demand", "expansion"]
NEGATIVE_KEYWORDS = ["miss", "decline", "loss", "cut", "downgrade", "bearish", "fall",
    "weak", "concern", "risk", "layoff", "penalty", "lawsuit", "drop",
    "sell", "shortage", "warning", "recall", "fraud"]


def keyword_sentiment(text: str) -> float:
    t = text.lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in t)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in t)
    if pos > neg:
        return 1.0
    elif neg > pos:
        return 0.0
    return 0.5


def retrieve_top_k(query: str, ticker: str, k=3, index_path=FAISS_INDEX_PATH,
                   articles=None) -> list:
    if articles:
        index, metadata = embed_articles(articles, ticker, index_path)
    else:
        index, metadata = load_faiss_index(ticker, index_path)

    if index is None or index.ntotal == 0:
        return []

    query_vector = embed_query(query)
    k_eff = min(k, index.ntotal)
    distances, indices = index.search(query_vector, k_eff)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        art = metadata[idx].copy()
        art["score_distance"] = float(dist)
        text = f"{art.get('title','')} {art.get('summary','')}"
        score = keyword_sentiment(text)
        art["sentiment"] = score
        art["sentiment_label"] = "Positif" if score == 1.0 else ("Négatif" if score == 0.0 else "Neutre")
        results.append(art)

    return results


def compute_news_sentiment_score(articles: list) -> float:
    if not articles:
        return 0.5
    return float(np.mean([a.get("sentiment", 0.5) for a in articles]))


def format_articles_for_llm(articles: list) -> str:
    if not articles:
        return "Aucune news récente disponible."
    parts = []
    for i, a in enumerate(articles, 1):
        summary = a.get("summary", "")[:200]
        if len(a.get("summary", "")) > 200:
            summary += "..."
        parts.append(
            f"[Article {i}] ({a.get('sentiment_label','Neutre')})\n"
            f"Titre  : {a.get('title','')}\n"
            f"Source : {a.get('source','')} — {a.get('published','')}\n"
            f"Résumé : {summary}"
        )
    return "\n\n".join(parts)

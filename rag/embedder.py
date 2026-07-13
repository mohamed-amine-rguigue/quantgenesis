import os
import sys
import logging
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Chargement modèle embedding : {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_articles(articles: list, ticker: str, index_path=FAISS_INDEX_PATH):
    ticker_dir = os.path.join(index_path, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    faiss_file = os.path.join(ticker_dir, "index.faiss")
    meta_file = os.path.join(ticker_dir, "metadata.pkl")

    if os.path.exists(faiss_file):
        index = faiss.read_index(faiss_file)
        with open(meta_file, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        metadata = []

    existing_links = {m["link"] for m in metadata}
    new_articles = [a for a in articles if a.get("link", "") not in existing_links]

    if not new_articles:
        return index, metadata

    model = get_embedding_model()
    texts = [f"{a.get('title','')} {a.get('summary','')}".strip() for a in new_articles]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    index.add(embeddings)
    metadata.extend(new_articles)

    faiss.write_index(index, faiss_file)
    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"[{ticker}] FAISS : {index.ntotal} vecteurs")
    return index, metadata


def load_faiss_index(ticker: str, index_path=FAISS_INDEX_PATH):
    faiss_file = os.path.join(index_path, ticker, "index.faiss")
    meta_file = os.path.join(index_path, ticker, "metadata.pkl")
    if not os.path.exists(faiss_file):
        return None, []
    index = faiss.read_index(faiss_file)
    with open(meta_file, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def embed_query(query: str) -> np.ndarray:
    model = get_embedding_model()
    return model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

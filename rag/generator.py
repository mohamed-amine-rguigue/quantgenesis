import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from rag.retriever import retrieve_top_k, compute_news_sentiment_score, format_articles_for_llm

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

SIGNAL_LABELS = {2: "HAUSSIER 📈", 1: "NEUTRE ➡️", 0: "BAISSIER 📉"}
SIGNAL_LABELS_RAW = {2: "haussier", 1: "neutre", 0: "baissier"}

PROMPT_TEMPLATE = """Tu es un analyste financier expert en semi-conducteurs. Réponds en français.

=== DONNÉES ===
Ticker        : {ticker}
Signal ML     : {signal_label}
Confiance ML  : {confidence_pct:.1f}%
Score d'achat : {buy_score:.0f}/100

=== ACTUALITÉS RÉCENTES ===
{news_context}

=== ANALYSE ===
**Synthèse du signal technique :**
[2-3 phrases sur le signal ML et sa confiance]

**Analyse des actualités :**
[3-4 phrases sur l'impact des news]

**Convergence signal / actualités :**
[2-3 phrases sur la convergence ou divergence]

**Points de vigilance :**
[1-2 risques à surveiller]

**Conclusion :**
[1-2 phrases de synthèse]

⚠️ Cette analyse est éducative uniquement — pas un conseil financier.
"""


def compute_buy_score(ml_confidence: float, news_sentiment: float) -> float:
    score = ml_confidence * 100 * 0.6 + news_sentiment * 100 * 0.4
    return min(max(score, 0.0), 100.0)


def generate_analysis(ticker: str, ml_class: int, ml_probas: list,
                      articles: list, index_path="faiss_index") -> dict:
    ml_confidence = float(ml_probas[ml_class]) if ml_probas else 0.5
    signal_label = SIGNAL_LABELS.get(ml_class, "NEUTRE ➡️")
    signal_raw = SIGNAL_LABELS_RAW.get(ml_class, "neutre")

    query = f"{ticker} stock {signal_raw} trend semiconductors earnings revenue AI chip"
    retrieved = retrieve_top_k(query, ticker, k=3, index_path=index_path, articles=articles)
    news_sentiment = compute_news_sentiment_score(retrieved)
    buy_score = compute_buy_score(ml_confidence, news_sentiment)
    news_context = format_articles_for_llm(retrieved)

    analysis_text = _generate_llm(ticker, signal_label, ml_confidence * 100, buy_score, news_context)

    return {
        "signal": signal_label,
        "signal_raw": signal_raw,
        "confidence": ml_confidence,
        "buy_score": buy_score,
        "news_sentiment": news_sentiment,
        "retrieved_articles": retrieved,
        "analysis_text": analysis_text,
    }


def _generate_llm(ticker, signal_label, confidence_pct, buy_score, news_context) -> str:
    try:
        llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)
        prompt = PromptTemplate(
            input_variables=["ticker", "signal_label", "confidence_pct", "buy_score", "news_context"],
            template=PROMPT_TEMPLATE,
        )
        # LCEL : prompt | llm (remplace LLMChain déprécié)
        chain = prompt | llm
        logger.info(f"[{ticker}] Génération LLM ({OLLAMA_MODEL})...")
        result = chain.invoke({"ticker": ticker, "signal_label": signal_label,
                               "confidence_pct": confidence_pct, "buy_score": buy_score,
                               "news_context": news_context})
        return result.strip() if isinstance(result, str) else str(result).strip()
    except Exception as e:
        logger.error(f"Erreur Ollama : {e}")
        return (
            f"**Analyse automatique — {ticker}**\n\n"
            f"Signal ML : {signal_label} ({confidence_pct:.1f}% de confiance)\n"
            f"Score d'achat : {buy_score:.0f}/100\n\n"
            f"**Actualités :**\n{news_context}\n\n"
            f"_Analyse LLM indisponible — lancez Ollama avec `ollama serve` et `ollama pull mistral`_\n\n"
            f"⚠️ Ceci n'est pas un conseil financier."
        )

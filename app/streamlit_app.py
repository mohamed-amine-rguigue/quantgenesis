import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT_DIR, ".env"))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="QuantGenesis — Semi-conducteurs",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #fafafa; }
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
    border: 1px solid #00d4aa;
}
.signal-card { padding: 1.5rem; border-radius: 12px; text-align: center; margin: 0.5rem 0; }
.signal-bullish { background: linear-gradient(135deg,#0d4f1c,#1a7a30); border: 2px solid #00e676; box-shadow: 0 0 20px rgba(0,230,118,.3); }
.signal-bearish { background: linear-gradient(135deg,#4f0d0d,#7a1a1a); border: 2px solid #ff5252; box-shadow: 0 0 20px rgba(255,82,82,.3); }
.signal-neutral { background: linear-gradient(135deg,#1a1a1a,#2a2a2a); border: 2px solid #ffd740; box-shadow: 0 0 20px rgba(255,215,64,.3); }
.news-card { background: #1a1a2e; border-left: 4px solid #00d4aa; padding: 1rem; border-radius: 4px; margin: 0.5rem 0; }
.news-positive { border-left-color: #00e676; }
.news-negative { border-left-color: #ff5252; }
.news-neutral  { border-left-color: #ffd740; }
.disclaimer { background: rgba(255,82,82,.1); border: 1px solid #ff5252; border-radius: 8px; padding: 1rem; margin-top: 2rem; font-size: .85rem; color: #ff8a80; }
.stButton > button {
    background: linear-gradient(135deg,#00d4aa,#0097a7); color: white;
    border: none; border-radius: 8px; padding: .75rem 2rem;
    font-size: 1rem; font-weight: bold; width: 100%;
}
</style>
""", unsafe_allow_html=True)

TICKERS = ["NVDA", "ASML", "INTC", "AMD", "TSM"]
TICKER_NAMES = {
    "NVDA": "NVIDIA Corporation", "ASML": "ASML Holding N.V.",
    "INTC": "Intel Corporation", "AMD": "Advanced Micro Devices",
    "TSM": "Taiwan Semiconductor",
}
DB_PATH = os.getenv("SQLITE_DB_PATH", os.path.join(ROOT_DIR, "quantgenesis.db"))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(ROOT_DIR, "models/trend_classifier.pt"))
SCALER_PATH = os.getenv("SCALER_PATH", os.path.join(ROOT_DIR, "models/scaler.pkl"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join(ROOT_DIR, "faiss_index"))


@st.cache_resource(show_spinner="Chargement du modèle ML...")
def load_ml_model():
    from ml.model import load_model
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    try:
        model = load_model(MODEL_PATH, device="cpu")
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        logger.error(f"Erreur chargement modèle : {e}")
        return None, None


@st.cache_data(ttl=3600, show_spinner="Récupération des données de marché...")
def get_market_data(ticker: str):
    from data.pipeline import init_db, fetch_prices, load_prices, fetch_news, load_news
    from data.features import build_features
    conn = init_db(DB_PATH)
    fetch_prices(ticker, conn)
    prices_df = load_prices(ticker, conn, days=365)
    articles = fetch_news(ticker, conn, max_articles=20)
    if not articles:
        articles = load_news(ticker, conn, limit=30)
    conn.close()
    features_df = build_features(prices_df) if not prices_df.empty else pd.DataFrame()
    return prices_df, features_df, articles


def predict_trend(features_df, model, scaler):
    from ml.dataset import prepare_inference_input
    if features_df.empty or model is None:
        return 1, [0.25, 0.40, 0.35], 0.40
    try:
        x = prepare_inference_input(features_df, scaler)
        probas_t, classe_t = model.predict(x)
        probas = probas_t[0].tolist()
        classe = int(classe_t[0].item())
        bull_prob = probas[2] if len(probas) > 2 else 0.0
        bear_prob = probas[0] if len(probas) > 0 else 0.0
        ml_signal_score = 50.0 + 50.0 * (bull_prob - bear_prob)
        return classe, probas, ml_signal_score
    except Exception as e:
        logger.error(f"Erreur inférence : {e}")
        return 1, [0.25, 0.40, 0.35], 0.40


def render_price_chart(prices_df, ticker):
    if prices_df.empty:
        st.warning("Données de prix indisponibles")
        return
    df = prices_df.tail(90).copy()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Prix", increasing_line_color="#00e676", decreasing_line_color="#ff5252",
        increasing_fillcolor="#00e676", decreasing_fillcolor="#ff5252",
    ))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma20"], mode="lines", name="MM 20j",
                             line=dict(color="#00d4aa", width=1.5)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma50"], mode="lines", name="MM 50j",
                             line=dict(color="#ffd740", width=1.5, dash="dash")))
    fig.update_layout(
        title=dict(text=f"{ticker} — 90 derniers jours", font=dict(size=18, color="#fafafa")),
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1a2e", font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#2a2a3e", rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="#2a2a3e", title="Prix (USD)"),
        legend=dict(bgcolor="#1a1a2e", bordercolor="#2a2a3e", borderwidth=1),
        height=450, margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_signal_gauge(classe, probas):
    labels = ["Baissier", "Neutre", "Haussier"]
    colors = ["#ff5252", "#ffd740", "#00e676"]
    fig = go.Figure()
    for i, (label, color, proba) in enumerate(zip(labels, colors, probas)):
        fig.add_trace(go.Bar(
            x=[proba * 100], y=[label], orientation="h",
            marker=dict(color=color, opacity=1.0 if i == classe else 0.35,
                        line=dict(color=color, width=3 if i == classe else 1)),
            text=f"{proba*100:.1f}%", textposition="inside",
            textfont=dict(size=14, color="white"), name=label, showlegend=False,
        ))
    fig.update_layout(
        title=dict(text="Probabilités par scénario", font=dict(size=14, color="#fafafa")),
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1a2e", font=dict(color="#fafafa"),
        xaxis=dict(range=[0, 100], gridcolor="#2a2a3e", title="Probabilité (%)"),
        yaxis=dict(gridcolor="#2a2a3e"),
        height=200, margin=dict(l=10, r=10, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_buy_score(score):
    color = "#00e676" if score >= 65 else ("#ffd740" if score >= 40 else "#ff5252")
    zone = "Zone favorable" if score >= 65 else ("Zone neutre" if score >= 40 else "Zone défavorable")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="/100", font=dict(size=36, color="#fafafa")),
        title=dict(text=f"Score d'achat<br><span style='font-size:.8em;color:{color}'>{zone}</span>",
                   font=dict(size=14, color="#fafafa")),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#fafafa")),
            bar=dict(color=color, thickness=0.3),
            bgcolor="#1a1a2e", borderwidth=2, bordercolor="#2a2a3e",
            steps=[
                dict(range=[0, 40], color="#2d0a0a"),
                dict(range=[40, 65], color="#1a1a0a"),
                dict(range=[65, 100], color="#0a2d1a"),
            ],
        ),
    ))
    fig.update_layout(paper_bgcolor="#0e1117", font=dict(color="#fafafa"),
                      height=280, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_news_cards(articles):
    if not articles:
        st.info("Aucune actualité disponible")
        return
    for a in articles:
        sentiment = a.get("sentiment_label", "Neutre")
        css = {"Positif": "news-positive", "Négatif": "news-negative", "Neutre": "news-neutral"}.get(sentiment, "news-neutral")
        emoji = {"Positif": "🟢", "Négatif": "🔴", "Neutre": "🟡"}.get(sentiment, "🟡")
        st.markdown(f"""
        <div class="news-card {css}">
            <div style="font-size:.85rem;color:#888;margin-bottom:4px">
                {emoji} {sentiment} &nbsp;|&nbsp; {a.get('source','')} &nbsp;|&nbsp; {str(a.get('published',''))[:16]}
            </div>
            <div style="font-size:1rem;font-weight:bold">
                <a href="{a.get('link','#')}" target="_blank" style="color:#fafafa;text-decoration:none">
                    {a.get('title','')}
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)


def main():
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;font-size:2.5rem;color:#00d4aa">⚡ QuantGenesis</h1>
        <p style="margin:.5rem 0 0 0;color:#aaa;font-size:1.1rem">
            Système d'analyse d'achat — Secteur Semi-conducteurs
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        ticker = st.selectbox(
            "Choisissez un ticker",
            options=TICKERS,
            format_func=lambda t: f"{t} — {TICKER_NAMES.get(t,'')}",
        )
        st.markdown("---")
        st.markdown("### 🛠️ Statut système")
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            st.success("✅ Modèle ML disponible")
        else:
            st.warning("⚠️ Modèle non entraîné\n\n`python ml/train.py`")
        try:
            import requests
            r = requests.get(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), timeout=2)
            if r.status_code == 200:
                st.success("✅ Ollama disponible")
            else:
                st.warning("⚠️ Ollama inaccessible")
        except Exception:
            st.warning("⚠️ Ollama hors ligne\n\n`ollama serve`")

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        analyze = st.button(f"🔍 Analyser {ticker}", use_container_width=True)

    if analyze or st.session_state.get("last_ticker"):
        if analyze:
            st.session_state["last_ticker"] = ticker
            if st.session_state.get("cached_ticker") != ticker:
                st.cache_data.clear()
                st.session_state["cached_ticker"] = ticker
        else:
            ticker = st.session_state.get("last_ticker", ticker)

        with st.spinner(f"Récupération des données {ticker}..."):
            try:
                prices_df, features_df, articles = get_market_data(ticker)
            except Exception as e:
                st.error(f"Erreur données : {e}")
                return

        model, scaler = load_ml_model()
        with st.spinner("Inférence ML..."):
            ml_class, ml_probas, ml_confidence = predict_trend(features_df, model, scaler)

        with st.spinner("Génération analyse RAG + LLM..."):
            try:
                from rag.generator import generate_analysis
                rag = generate_analysis(ticker, ml_class, ml_probas, articles, FAISS_INDEX_PATH)
            except Exception as e:
                logger.error(f"Erreur RAG : {e}")
                from rag.generator import SIGNAL_LABELS, SIGNAL_LABELS_RAW, compute_buy_score
                rag = {
                    "signal": SIGNAL_LABELS.get(ml_class, "NEUTRE ➡️"),
                    "signal_raw": SIGNAL_LABELS_RAW.get(ml_class, "neutre"),
                    "confidence": ml_confidence,
                    "buy_score": compute_buy_score(ml_confidence, 0.5),
                    "news_sentiment": 0.5,
                    "retrieved_articles": [],
                    "analysis_text": f"Erreur analyse : {e}",
                }

        signal_raw = rag["signal_raw"]
        SIGNAL_CSS = {"haussier": "signal-bullish", "neutre": "signal-neutral", "baissier": "signal-bearish"}
        SIGNAL_COLORS = {"haussier": "#00e676", "neutre": "#ffd740", "baissier": "#ff5252"}

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            css = SIGNAL_CSS.get(signal_raw, "signal-neutral")
            color = SIGNAL_COLORS.get(signal_raw, "#ffd740")
            st.markdown(f"""
            <div class="signal-card {css}">
                <div style="font-size:2.5rem;margin-bottom:.5rem">{rag['signal']}</div>
                <div style="font-size:1.2rem;color:{color}">
                    Confiance : {rag['confidence']*100:.1f}%
                </div>
                <div style="font-size:.9rem;color:#aaa;margin-top:.5rem">Prédiction à 5 jours</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            render_buy_score(rag["buy_score"])

        with col3:
            if not prices_df.empty:
                last = prices_df["close"].iloc[-1]
                prev = prices_df["close"].iloc[-2] if len(prices_df) > 1 else last
                d1 = (last - prev) / prev * 100
                w = prices_df["close"].iloc[-6] if len(prices_df) > 5 else prev
                dw = (last - w) / w * 100
                h52 = prices_df["close"].tail(252).max()
                l52 = prices_df["close"].tail(252).min()
                st.markdown(f"#### {TICKER_NAMES.get(ticker, ticker)}")
                c1, c2 = st.columns(2)
                c1.metric("Cours actuel", f"${last:.2f}", f"{d1:+.2f}%")
                c2.metric("Variation 1 sem.", f"{dw:+.2f}%")
                c3, c4 = st.columns(2)
                c3.metric("Plus haut 52 sem.", f"${h52:.2f}")
                c4.metric("Plus bas 52 sem.", f"${l52:.2f}")

        st.markdown("---")
        render_price_chart(prices_df, ticker)

        col_proba, col_ind = st.columns(2)
        with col_proba:
            render_signal_gauge(ml_class, ml_probas)
        with col_ind:
            if not features_df.empty:
                st.markdown("#### 📊 Indicateurs techniques")
                r = features_df.iloc[-1]
                a1, a2 = st.columns(2)
                rsi_val = r.get("rsi", 0)
                rsi_delta = "Suracheté" if rsi_val > 70 else ("Survendu" if rsi_val < 30 else "Neutre")
                a1.metric("RSI (14j)", f"{rsi_val:.1f}", rsi_delta)
                a2.metric("MACD", f"{r.get('macd',0):.4f}", f"Signal: {r.get('macd_signal',0):.4f}")
                a3, a4 = st.columns(2)
                a3.metric("Dist. MM 20j", f"{r.get('price_vs_sma20',0)*100:+.2f}%")
                a4.metric("Dist. MM 50j", f"{r.get('price_vs_sma50',0)*100:+.2f}%")
                a5, a6 = st.columns(2)
                a5.metric("Volatilité 20j", f"{r.get('volatility',0)*100:.1f}%")
                a6.metric("Volume z-score", f"{r.get('volume_zscore',0):.2f}")

        st.markdown("---")
        st.markdown("### 🤖 Analyse IA (Mistral via Ollama)")
        analysis_html = rag["analysis_text"].replace("\n", "<br>")
        st.markdown(
            f'<div style="background:#1a1a2e;border:1px solid #00d4aa;border-radius:12px;'
            f'padding:1.5rem;line-height:1.8">{analysis_html}</div>',
            unsafe_allow_html=True
        )

        retrieved = rag.get("retrieved_articles", [])
        if retrieved:
            st.markdown("---")
            st.markdown("### 📰 Sources utilisées dans l'analyse")
            render_news_cards(retrieved)
        elif articles:
            st.markdown("---")
            st.markdown("### 📰 Dernières actualités")
            render_news_cards(articles[:5])

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>AVERTISSEMENT LÉGAL</strong> — Cette application est fournie à des fins
            éducatives uniquement. Les analyses présentées <strong>ne constituent pas des conseils
            financiers ou d'investissement</strong>. Consultez un conseiller financier agréé
            avant tout investissement.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem;color:#555">
            <div style="font-size:5rem">⚡</div>
            <h2 style="color:#00d4aa">Bienvenue sur QuantGenesis</h2>
            <p style="font-size:1.1rem">
                Sélectionnez un ticker et cliquez sur
                <strong style="color:#00d4aa">Analyser</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

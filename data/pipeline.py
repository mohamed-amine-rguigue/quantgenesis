import sqlite3
import os
import time
import logging
from datetime import datetime, timedelta

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import yfinance as yf
import feedparser
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("SQLITE_DB_PATH", "quantgenesis.db")
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", 1000))
DEFAULT_TICKERS = [t.strip() for t in os.getenv(
    "TICKERS",
    "NVDA,ASML,INTC,AMD,TSM,AVGO,ARM,QCOM,TXN,ADI"
).split(",") if t.strip()]

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "https://www.investing.com/rss/news.rss",
]


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            UNIQUE(ticker, date)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT, link TEXT, published TEXT, source TEXT,
            inserted_at TEXT DEFAULT (datetime('now')),
            UNIQUE(ticker, link)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            title TEXT,
            sentiment_score REAL,
            relevance_score REAL,
            source TEXT,
            inserted_at TEXT DEFAULT (datetime('now')),
            UNIQUE(ticker, date, title)
        )
    """)
    conn.commit()
    return conn


def _has_cached_prices(ticker: str, conn: sqlite3.Connection, days: int = HISTORY_DAYS) -> bool:
    since = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM prices WHERE ticker=? AND date>=?", (ticker, since))
    return cursor.fetchone()[0] > 0


def fetch_prices(ticker: str, conn: sqlite3.Connection, days: int = HISTORY_DAYS) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    if _has_cached_prices(ticker, conn, days):
        logger.info(f"[{ticker}] Données locales déjà présentes, utilisation du cache SQLite")
        return load_prices(ticker, conn, days)

    logger.info(f"[{ticker}] Téléchargement des prix depuis Yahoo Finance...")
    try:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=20,
        )
    except Exception as e:
        logger.warning(f"[{ticker}] Yahoo Finance indisponible : {e}")
        return pd.DataFrame()

    if df.empty:
        logger.warning(f"[{ticker}] Aucune donnée retournée par Yahoo Finance")
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df["ticker"] = ticker
    df["date"] = df["date"].astype(str)

    cursor = conn.cursor()
    rows_inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO prices (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ticker, row["date"], float(row.get("open", 0)), float(row.get("high", 0)),
                  float(row.get("low", 0)), float(row.get("close", 0)), float(row.get("volume", 0))))
            rows_inserted += cursor.rowcount
        except Exception:
            pass
    conn.commit()
    logger.info(f"[{ticker}] {rows_inserted} nouvelles lignes insérées")
    return df


def load_prices(ticker: str, conn: sqlite3.Connection, days: int = HISTORY_DAYS) -> pd.DataFrame:
    since = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM prices WHERE ticker=? AND date>=? ORDER BY date ASC",
        conn, params=(ticker, since)
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_news_sentiment_history(ticker: str, conn: sqlite3.Connection,
                                 time_from: str = "20220101T0000",
                                 limit: int = 1000) -> int:
    """Backfill un historique daté de sentiment news via Alpha Vantage NEWS_SENTIMENT.
    Contrairement aux flux RSS (actualité du jour seulement), cet endpoint renvoie des
    articles avec une date de publication passée, ce qui permet d'aligner un vrai signal
    de sentiment sur l'historique de prix utilisé pour l'entraînement (pas de fuite :
    seul le sentiment antérieur ou égal à chaque date de feature est utilisé)."""
    if not ALPHAVANTAGE_API_KEY:
        logger.warning(f"[{ticker}] ALPHAVANTAGE_API_KEY absente — sentiment historique non récupéré")
        return 0

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "limit": limit,
        "sort": "EARLIEST",
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    try:
        resp = requests.get(ALPHAVANTAGE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        logger.exception(f"[{ticker}] Erreur Alpha Vantage")
        return 0

    feed = payload.get("feed", [])
    if not feed:
        logger.warning(f"[{ticker}] Aucun article renvoyé par Alpha Vantage "
                       f"({payload.get('Information') or payload.get('Note') or 'réponse vide'})")
        return 0

    cursor = conn.cursor()
    inserted = 0
    for article in feed:
        time_published = article.get("time_published", "")
        if len(time_published) < 8:
            continue
        date = f"{time_published[0:4]}-{time_published[4:6]}-{time_published[6:8]}"
        title = article.get("title", "")
        source = article.get("source", "")

        ticker_sentiment = next(
            (ts for ts in article.get("ticker_sentiment", []) if ts.get("ticker") == ticker),
            None,
        )
        if ticker_sentiment:
            score = float(ticker_sentiment.get("ticker_sentiment_score", 0.0))
            relevance = float(ticker_sentiment.get("relevance_score", 0.0))
        else:
            score = float(article.get("overall_sentiment_score", 0.0))
            relevance = 0.0

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO news_sentiment
                (ticker, date, title, sentiment_score, relevance_score, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ticker, date, title, score, relevance, source))
            inserted += cursor.rowcount
        except Exception:
            pass
    conn.commit()
    logger.info(f"[{ticker}] {inserted} entrées de sentiment historique insérées")
    return inserted


def load_news_sentiment_daily(ticker: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Agrège le sentiment par jour, pondéré par la pertinence de l'article pour le ticker."""
    df = pd.read_sql_query("""
        SELECT date, sentiment_score, relevance_score FROM news_sentiment WHERE ticker=?
    """, conn, params=(ticker,))
    if df.empty:
        return pd.DataFrame(columns=["date", "news_sentiment"])

    df["weight"] = df["relevance_score"].clip(lower=0.01)
    daily = (
        df.groupby("date")
        .apply(lambda g: float(np.average(g["sentiment_score"], weights=g["weight"])))
        .reset_index(name="news_sentiment")
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def fetch_news(ticker: str, conn: sqlite3.Connection, max_articles: int = 30) -> list:
    articles = []
    cursor = conn.cursor()
    for feed_template in RSS_FEEDS:
        url = feed_template.format(ticker=ticker)
        logger.info(f"[{ticker}] RSS : {url[:60]}...")
        try:
            feed = feedparser.parse(url)
            entries = feed.entries[:max_articles]
        except Exception as e:
            logger.warning(f"Erreur feedparser : {e}")
            continue
        for entry in entries:
            title = getattr(entry, "title", "").strip()
            summary = getattr(entry, "summary", "").strip()
            link = getattr(entry, "link", "").strip()
            published = getattr(entry, "published", str(datetime.now()))
            source = feed.feed.get("title", url)
            if not title or not link:
                continue
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO news (ticker, title, summary, link, published, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (ticker, title, summary, link, published, source))
                conn.commit()
            except Exception:
                pass
            articles.append({"ticker": ticker, "title": title, "summary": summary,
                              "link": link, "published": published, "source": source})
        time.sleep(0.5)
    logger.info(f"[{ticker}] {len(articles)} articles récupérés")
    return articles


def load_news(ticker: str, conn: sqlite3.Connection, limit: int = 50) -> list:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT title, summary, link, published, source FROM news
        WHERE ticker=? ORDER BY published DESC LIMIT ?
    """, (ticker, limit))
    return [{"title": r[0], "summary": r[1], "link": r[2], "published": r[3], "source": r[4]}
            for r in cursor.fetchall()]


def run_pipeline(ticker: str, db_path: str = DB_PATH) -> dict:
    conn = init_db(db_path)
    prices_df = fetch_prices(ticker, conn)
    if prices_df.empty:
        prices_df = load_prices(ticker, conn)
    articles = fetch_news(ticker, conn)
    if not articles:
        articles = load_news(ticker, conn)
    conn.close()
    return {"ticker": ticker, "prices": prices_df, "news": articles}


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    result = run_pipeline(ticker)
    print(f"Ticker : {result['ticker']}")
    print(f"Lignes de prix : {len(result['prices'])}")
    print(f"Articles news  : {len(result['news'])}")

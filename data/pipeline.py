import sqlite3
import os
import time
import logging
from datetime import datetime, timedelta

import yfinance as yf
import feedparser
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("SQLITE_DB_PATH", "quantgenesis.db")
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", 365))

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
    conn.commit()
    return conn


def fetch_prices(ticker: str, conn: sqlite3.Connection, days: int = HISTORY_DAYS) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    logger.info(f"[{ticker}] Téléchargement des prix...")
    try:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        logger.error(f"[{ticker}] Erreur yfinance : {e}")
        return pd.DataFrame()

    if df.empty:
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


def load_prices(ticker: str, conn: sqlite3.Connection, days: int = 365) -> pd.DataFrame:
    since = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM prices WHERE ticker=? AND date>=? ORDER BY date ASC",
        conn, params=(ticker, since)
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


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

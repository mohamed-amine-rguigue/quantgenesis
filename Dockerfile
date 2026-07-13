FROM python:3.11-slim

LABEL description="QuantGenesis — Analyse Semi-conducteurs"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/models /app/faiss_index /app/mlruns

RUN cp .env.example .env 2>/dev/null || true

# Pré-téléchargement du modèle sentence-transformers
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    || echo "Téléchargement différé au premier lancement"

RUN mkdir -p /root/.streamlit && printf '[server]\nheadless = true\naddress = "0.0.0.0"\nport = 8501\nenableCORS = false\n\n[theme]\nbase = "dark"\nprimaryColor = "#00d4aa"\nbackgroundColor = "#0e1117"\nsecondaryBackgroundColor = "#1a1a2e"\ntextColor = "#fafafa"\n' > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]

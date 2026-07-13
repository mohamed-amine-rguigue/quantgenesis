import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.pipeline import init_db, fetch_prices, load_prices
from data.features import build_features
from ml.dataset import build_datasets, create_dataloaders
from ml.model import TrendClassifier, save_model, count_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_HP = {
    "epochs": 50, "batch_size": 64, "learning_rate": 1e-3,
    "weight_decay": 1e-4, "dropout_rates": [0.3, 0.2, 0.0],
    "hidden_dims": [128, 64, 32], "val_ratio": 0.2,
    "patience": 10, "scheduler_patience": 5, "scheduler_factor": 0.5,
}

MODEL_PATH = os.getenv("MODEL_PATH", "models/trend_classifier.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item()
            preds.extend(logits.argmax(-1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return total_loss / max(len(loader), 1), accuracy_score(labels, preds), np.array(labels), np.array(preds)


def train(tickers=["NVDA", "ASML", "INTC", "AMD", "TSM"], hyperparams=None, db_path="quantgenesis.db"):
    if hyperparams is None:
        hyperparams = DEFAULT_HP.copy()

    import pandas as pd
    conn = init_db(db_path)
    all_features = []
    for ticker in tickers:
        logger.info(f"Chargement {ticker}...")
        fetch_prices(ticker, conn)
        prices = load_prices(ticker, conn)
        if prices.empty:
            continue
        feat = build_features(prices)
        if not feat.empty:
            all_features.append(feat)
    conn.close()

    if not all_features:
        raise RuntimeError("Aucune donnée pour l'entraînement")

    df = pd.concat(all_features, ignore_index=True)
    os.makedirs("models", exist_ok=True)

    train_ds, val_ds, _ = build_datasets(df, hyperparams["val_ratio"], SCALER_PATH)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, hyperparams["batch_size"])

    device = "cpu"
    model = TrendClassifier(hidden_dims=hyperparams["hidden_dims"],
                             dropout_rates=hyperparams["dropout_rates"]).to(device)
    logger.info(f"Modèle : {count_parameters(model):,} paramètres")

    optimizer = Adam(model.parameters(), lr=hyperparams["learning_rate"],
                     weight_decay=hyperparams["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  patience=hyperparams["scheduler_patience"],
                                  factor=hyperparams["scheduler_factor"])

    counts = np.bincount(train_ds.y.numpy(), minlength=3).astype(float)
    weights = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # MLflow — si indisponible on continue sans tracking
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("QuantGenesis-TrendClassifier")
        run = mlflow.start_run()
        mlflow.log_params({**hyperparams, "tickers": ",".join(tickers),
                           "n_train": len(train_ds), "n_val": len(val_ds)})
        use_mlflow = True
    except Exception:
        run = None
        use_mlflow = False
        logger.warning("MLflow indisponible — tracking désactivé")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, hyperparams["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        logger.info(f"Époque {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f}")
        if use_mlflow:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}, step=epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, MODEL_PATH)
            logger.info(f"  → Meilleur modèle (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= hyperparams["patience"]:
                logger.info(f"Early stopping à l'époque {epoch}")
                break

    _, final_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    report = classification_report(y_true, y_pred, target_names=["Baissier", "Neutre", "Haussier"], zero_division=0)
    logger.info(f"\n{report}")

    if use_mlflow:
        mlflow.log_metric("final_val_accuracy", final_acc)
        mlflow.log_text(report, "classification_report.txt")
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()

    logger.info("Entraînement terminé.")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["NVDA", "ASML", "INTC", "AMD", "TSM"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    hp = DEFAULT_HP.copy()
    hp["epochs"] = args.epochs
    hp["batch_size"] = args.batch_size
    hp["learning_rate"] = args.lr
    train(tickers=args.tickers, hyperparams=hp)
    print(f"Modèle sauvegardé : {MODEL_PATH}")

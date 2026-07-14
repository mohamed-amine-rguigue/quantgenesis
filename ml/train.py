import os
import sys
import logging
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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

from data.pipeline import init_db, fetch_prices, load_prices, DEFAULT_TICKERS
from data.features import build_features
from ml.dataset import build_datasets, build_datasets_from_frames, create_dataloaders
from ml.model import TrendClassifier, save_model, count_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

DEFAULT_HP = {
    "epochs": 150, "batch_size": 64, "learning_rate": 5e-4,
    "weight_decay": 5e-4, "dropout_rates": [0.4, 0.3, 0.2],
    "hidden_dims": [64, 32, 16], "val_ratio": 0.2,
    "patience": 15, "scheduler_patience": 5, "scheduler_factor": 0.5,
}

MODEL_PATH = os.getenv("MODEL_PATH", "models/trend_classifier.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")


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


def load_feature_dataframe(tickers, db_path="quantgenesis.db"):
    """Charge prix + features techniques pour une liste de tickers, et retourne un
    DataFrame concaténé trié par date. Le sentiment news (voir data/pipeline.py
    fetch_news_sentiment_history) n'est pas injecté ici : couverture historique
    gratuite trop incomplète pour être un signal fiable à l'entraînement
    (cf. FEATURE_COLS dans data/features.py)."""
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
        raise RuntimeError("Aucune donnée disponible")

    df = pd.concat(all_features, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def train(tickers=None, hyperparams=None, db_path="quantgenesis.db"):
    if tickers is None:
        tickers = DEFAULT_TICKERS
    if hyperparams is None:
        hyperparams = DEFAULT_HP.copy()

    df = load_feature_dataframe(tickers, db_path)
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
        mlflow.start_run()
        mlflow.log_params({**hyperparams, "tickers": ",".join(tickers),
                           "n_train": len(train_ds), "n_val": len(val_ds)})
        use_mlflow = True
    except Exception:
        use_mlflow = False
        logger.warning("MLflow indisponible — tracking désactivé")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, hyperparams["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, train_acc, _, _ = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        gap = train_acc - val_acc
        logger.info(f"Époque {epoch:03d} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} "
                   f"| Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | Gap: {gap:+.4f}")
        if use_mlflow:
            mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_acc,
                               "val_loss": val_loss, "val_accuracy": val_acc,
                               "train_val_gap": gap}, step=epoch)
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

    _, final_train_acc, y_true_train, y_pred_train = evaluate(model, train_loader, criterion, device)
    _, final_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)

    train_report = classification_report(y_true_train, y_pred_train,
                                        target_names=["Baissier", "Neutre", "Haussier"], zero_division=0)
    val_report = classification_report(y_true, y_pred, target_names=["Baissier", "Neutre", "Haussier"], zero_division=0)
    logger.info(f"\n=== Performance sur le train (accuracy={final_train_acc:.4f}) ===\n{train_report}")
    logger.info(f"\n=== Performance sur la validation (accuracy={final_acc:.4f}) ===\n{val_report}")
    logger.info(f"Écart train/val (surapprentissage si élevé) : {final_train_acc - final_acc:+.4f}")

    if use_mlflow:
        mlflow.log_metric("final_train_accuracy", final_train_acc)
        mlflow.log_metric("final_val_accuracy", final_acc)
        mlflow.log_metric("final_train_val_gap", final_train_acc - final_acc)
        mlflow.log_text(train_report, "train_classification_report.txt")
        mlflow.log_text(val_report, "val_classification_report.txt")
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()

    logger.info("Entraînement terminé.")
    return model


def walk_forward_evaluate(tickers=None, n_windows=5,
                          test_days=30, min_train_days=180, hyperparams=None,
                          db_path="quantgenesis.db", seed=42):
    """Évalue le modèle sur plusieurs fenêtres temporelles tirées au hasard dans le passé :
    pour chaque fenêtre, entraîne uniquement sur les données antérieures à une date de coupure
    et teste sur les `test_days` jours suivants, en comparant à une baseline naïve
    (classe majoritaire). Permet de voir si la performance dépend du régime de marché."""
    if tickers is None:
        tickers = DEFAULT_TICKERS
    if hyperparams is None:
        hyperparams = DEFAULT_HP.copy()
        hyperparams["epochs"] = 30
        hyperparams["patience"] = 5

    import pandas as pd
    df = load_feature_dataframe(tickers, db_path)

    min_date, max_date = df["date"].min(), df["date"].max()
    earliest_cutoff = min_date + pd.Timedelta(days=min_train_days)
    latest_cutoff = max_date - pd.Timedelta(days=test_days)

    if earliest_cutoff >= latest_cutoff:
        raise RuntimeError("Historique trop court pour une évaluation walk-forward "
                           "(augmenter HISTORY_DAYS ou réduire min_train_days/test_days)")

    rng = np.random.default_rng(seed)
    candidate_days = (latest_cutoff - earliest_cutoff).days
    offsets = rng.choice(candidate_days + 1, size=min(n_windows, candidate_days + 1), replace=False)
    cutoffs = sorted(earliest_cutoff + pd.Timedelta(days=int(d)) for d in offsets)

    device = "cpu"
    results = []

    for i, cutoff in enumerate(cutoffs, 1):
        train_df = df[df["date"] <= cutoff]
        test_df = df[(df["date"] > cutoff) & (df["date"] <= cutoff + pd.Timedelta(days=test_days))]

        if len(train_df) < 50 or len(test_df) < 10:
            logger.warning(f"Fenêtre {i} ignorée (données insuffisantes autour de {cutoff.date()})")
            continue

        scaler_path = f"models/scaler_wf_{i}.pkl"
        train_ds, test_ds, _ = build_datasets_from_frames(train_df, test_df, scaler_path)
        train_loader, test_loader = create_dataloaders(train_ds, test_ds, hyperparams["batch_size"])

        model = TrendClassifier(hidden_dims=hyperparams["hidden_dims"],
                                dropout_rates=hyperparams["dropout_rates"]).to(device)
        optimizer = Adam(model.parameters(), lr=hyperparams["learning_rate"],
                         weight_decay=hyperparams["weight_decay"])
        counts = np.bincount(train_ds.y.numpy(), minlength=3).astype(float)
        weights = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights)

        best_loss, patience_counter = float("inf"), 0
        for _ in range(1, hyperparams["epochs"] + 1):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, _, _, _ = evaluate(model, test_loader, criterion, device)
            if val_loss < best_loss:
                best_loss, patience_counter = val_loss, 0
            else:
                patience_counter += 1
                if patience_counter >= hyperparams["patience"]:
                    break

        _, train_acc, _, _ = evaluate(model, train_loader, criterion, device)
        _, acc, y_true, _ = evaluate(model, test_loader, criterion, device)
        baseline_acc = np.bincount(y_true, minlength=3).max() / len(y_true)
        gap = train_acc - acc

        logger.info(f"Fenêtre {i}: cutoff={cutoff.date()} | train={len(train_df)} | test={len(test_df)} "
                   f"| train_acc={train_acc:.3f} | test_acc={acc:.3f} | baseline={baseline_acc:.3f} "
                   f"| gap={gap:+.3f}")

        results.append({
            "window": i, "cutoff": cutoff.date().isoformat(),
            "n_train": len(train_df), "n_test": len(test_df),
            "train_accuracy": train_acc, "accuracy": acc, "baseline_accuracy": baseline_acc,
            "edge": acc - baseline_acc, "train_test_gap": gap,
        })

        if os.path.exists(scaler_path):
            os.remove(scaler_path)

    if not results:
        raise RuntimeError("Aucune fenêtre valide pour l'évaluation walk-forward")

    train_accs = [r["train_accuracy"] for r in results]
    accs = [r["accuracy"] for r in results]
    baselines = [r["baseline_accuracy"] for r in results]
    gaps = [r["train_test_gap"] for r in results]

    logger.info("\n=== Résumé walk-forward ===")
    for r in results:
        logger.info(f"  Fenêtre {r['window']} ({r['cutoff']}): train_acc={r['train_accuracy']:.3f} "
                   f"| test_acc={r['accuracy']:.3f} vs baseline={r['baseline_accuracy']:.3f} "
                   f"(edge={r['edge']:+.3f}, gap={r['train_test_gap']:+.3f})")
    logger.info(f"Train accuracy moyenne : {np.mean(train_accs):.3f} ± {np.std(train_accs):.3f}")
    logger.info(f"Test accuracy moyenne  : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    logger.info(f"Baseline moyenne       : {np.mean(baselines):.3f} ± {np.std(baselines):.3f}")
    logger.info(f"Edge moyen (test)      : {np.mean(accs) - np.mean(baselines):+.3f}")
    logger.info(f"Gap moyen (train-test) : {np.mean(gaps):+.3f} (élevé = surapprentissage)")

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("QuantGenesis-WalkForward")
        with mlflow.start_run():
            mlflow.log_params({"n_windows": len(results), "test_days": test_days,
                              "min_train_days": min_train_days, "tickers": ",".join(tickers)})
            for r in results:
                mlflow.log_metrics({f"train_acc_window_{r['window']}": r["train_accuracy"],
                                   f"test_acc_window_{r['window']}": r["accuracy"],
                                   f"baseline_window_{r['window']}": r["baseline_accuracy"]})
            mlflow.log_metric("mean_train_accuracy", float(np.mean(train_accs)))
            mlflow.log_metric("mean_test_accuracy", float(np.mean(accs)))
            mlflow.log_metric("mean_baseline_accuracy", float(np.mean(baselines)))
            mlflow.log_metric("mean_train_test_gap", float(np.mean(gaps)))
    except Exception:
        logger.warning("MLflow indisponible — tracking désactivé")

    return results


def train_with_walk_forward(tickers=None, n_windows=4,
                           test_days=30, min_train_days=180, hyperparams=None,
                           db_path="quantgenesis.db", seed=42):
    """Entraîne le modèle sur plusieurs fenêtres temporelles et retourne les résultats de validation."""
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
        raise RuntimeError("Aucune donnée pour l'évaluation")

    df = pd.concat(all_features, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    min_date, max_date = df["date"].min(), df["date"].max()
    earliest_cutoff = min_date + pd.Timedelta(days=min_train_days)
    latest_cutoff = max_date - pd.Timedelta(days=test_days)
    if earliest_cutoff >= latest_cutoff:
        raise RuntimeError("Historique trop court pour une validation walk-forward")

    rng = np.random.default_rng(seed)
    candidate_days = (latest_cutoff - earliest_cutoff).days
    offsets = rng.choice(candidate_days + 1, size=min(n_windows, candidate_days + 1), replace=False)
    cutoffs = sorted(earliest_cutoff + pd.Timedelta(days=int(d)) for d in offsets)

    results = []
    for i, cutoff in enumerate(cutoffs, 1):
        train_df = df[df["date"] <= cutoff]
        test_df = df[(df["date"] > cutoff) & (df["date"] <= cutoff + pd.Timedelta(days=test_days))]
        if len(train_df) < 80 or len(test_df) < 20:
            continue

        train_ds, test_ds, _ = build_datasets_from_frames(train_df, test_df, scaler_path=f"models/scaler_wf_{i}.pkl")
        train_loader, test_loader = create_dataloaders(train_ds, test_ds, hyperparams["batch_size"])

        device = "cpu"
        model = TrendClassifier(hidden_dims=hyperparams["hidden_dims"],
                                dropout_rates=hyperparams["dropout_rates"]).to(device)
        optimizer = Adam(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])
        counts = np.bincount(train_ds.y.numpy(), minlength=3).astype(float)
        weights = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights)

        best_loss, patience_counter = float("inf"), 0
        for epoch in range(1, hyperparams["epochs"] + 1):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, _, _, _ = evaluate(model, test_loader, criterion, device)
            if val_loss < best_loss:
                best_loss, patience_counter = val_loss, 0
            else:
                patience_counter += 1
                if patience_counter >= hyperparams["patience"]:
                    break

        _, train_acc, _, _ = evaluate(model, train_loader, criterion, device)
        _, acc, _, _ = evaluate(model, test_loader, criterion, device)
        results.append({"window": i, "cutoff": cutoff.date().isoformat(), "train_acc": train_acc, "test_acc": acc})
        logger.info(f"Fenêtre {i}: cutoff={cutoff.date()} | train_acc={train_acc:.3f} | test_acc={acc:.3f}")

    if not results:
        raise RuntimeError("Aucune fenêtre valide n’a pu être évaluée")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_HP["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_HP["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_HP["learning_rate"])
    parser.add_argument("--walk-forward", action="store_true",
                        help="Évalue sur plusieurs fenêtres temporelles au lieu d'un split unique")
    parser.add_argument("--n-windows", type=int, default=5)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--min-train-days", type=int, default=180)
    args = parser.parse_args()
    hp = DEFAULT_HP.copy()
    hp["epochs"] = args.epochs
    hp["batch_size"] = args.batch_size
    hp["learning_rate"] = args.lr

    if args.walk_forward:
        walk_forward_evaluate(tickers=args.tickers, n_windows=args.n_windows,
                             test_days=args.test_days, min_train_days=args.min_train_days,
                             hyperparams=hp)
    else:
        train(tickers=args.tickers, hyperparams=hp)
        print(f"Modèle sauvegardé : {MODEL_PATH}")

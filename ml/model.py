import os
import sys
import logging

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.features import FEATURE_COLS, NUM_CLASSES

INPUT_DIM = len(FEATURE_COLS)


class TrendClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dims=[128, 64, 32],
                 num_classes=NUM_CLASSES, dropout_rates=[0.3, 0.2, 0.0]):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = list(hidden_dims)
        self.dropout_rates = list(dropout_rates)
        layers = []
        in_dim = input_dim
        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers += [nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probas = F.softmax(logits, dim=-1)
            return probas, probas.argmax(dim=-1)


def save_model(model: TrendClassifier, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "num_classes": model.num_classes,
        "hidden_dims": getattr(model, "hidden_dims", None),
        "dropout_rates": getattr(model, "dropout_rates", None),
    }, path)
    logger.info(f"Modèle sauvegardé : {path}")


def load_model(path: str, device="cpu") -> TrendClassifier:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        hidden_dims = ckpt.get("hidden_dims", [128, 64, 32])
        dropout_rates = ckpt.get("dropout_rates", [0.3, 0.2, 0.0])
        model = TrendClassifier(input_dim=ckpt.get("input_dim", INPUT_DIM),
                                hidden_dims=hidden_dims,
                                num_classes=ckpt.get("num_classes", NUM_CLASSES),
                                dropout_rates=dropout_rates)
        try:
            model.load_state_dict(ckpt["state_dict"], strict=True)
        except RuntimeError as exc:
            logger.warning(f"Chargement strict impossible, tentative tolérante : {exc}")
            state = ckpt["state_dict"]
            model_state = model.state_dict()
            filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
            model_state.update(filtered)
            model.load_state_dict(model_state, strict=False)
    else:
        model = ckpt
    model.to(device).eval()
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

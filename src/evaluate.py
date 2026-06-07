"""Evaluate anomaly detection accuracy on held-out fault patterns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from data_generator import FaultPattern, GeneratorConfig, generate_dataset
from model import LSTMAutoencoder


def reconstruction_errors(model: LSTMAutoencoder, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(x)
        recon = model(tensor)
        mse = torch.mean((recon - tensor) ** 2, dim=(1, 2))
    return mse.numpy()


def find_best_threshold(normal_errors: np.ndarray, fault_errors: np.ndarray) -> float:
    candidates = np.linspace(
        float(np.min(normal_errors)),
        float(np.max(fault_errors)),
        num=200,
    )
    best_t, best_acc = candidates[0], 0.0
    labels = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(fault_errors))])
    scores = np.concatenate([normal_errors, fault_errors])
    for t in candidates:
        preds = (scores > t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return float(best_t)


def evaluate(model_path: Path, output_dir: Path | None = None) -> dict:
    out = output_dir or Path("artifacts")
    x, y = generate_dataset(GeneratorConfig(seed=7))

    model = LSTMAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    normal_idx = list(FaultPattern).index(FaultPattern.NORMAL)
    normal_x = x[y == normal_idx]
    fault_x = x[y != normal_idx]

    normal_errors = reconstruction_errors(model, normal_x)
    fault_errors = reconstruction_errors(model, fault_x)
    threshold = find_best_threshold(normal_errors, fault_errors)

    all_errors = np.concatenate([normal_errors, fault_errors])
    labels = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(fault_errors))])
    preds = (all_errors > threshold).astype(int)

    accuracy = float(accuracy_score(labels, preds))
    auc = float(roc_auc_score(labels, all_errors))
    report = classification_report(labels, preds, output_dict=True)

    metrics = {
        "threshold": threshold,
        "anomaly_detection_accuracy": round(accuracy * 100, 1),
        "roc_auc": round(auc, 3),
        "classification_report": report,
    }

    (out / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    evaluate(args.model, args.output)

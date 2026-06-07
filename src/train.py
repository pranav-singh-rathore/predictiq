"""Train LSTM autoencoder on normal operating data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_generator import FaultPattern, GeneratorConfig, generate_dataset
from model import LSTMAutoencoder


def train(
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    output_dir: Path | None = None,
) -> dict:
    out = output_dir or Path("artifacts")
    out.mkdir(parents=True, exist_ok=True)

    x, y = generate_dataset(GeneratorConfig())
    normal_mask = y == list(FaultPattern).index(FaultPattern.NORMAL)
    x_normal = x[normal_mask]

    split = int(0.85 * len(x_normal))
    train_x = torch.from_numpy(x_normal[:split])
    val_x = torch.from_numpy(x_normal[split:])

    train_loader = DataLoader(TensorDataset(train_x), batch_size=batch_size, shuffle=True)
    model = LSTMAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)

        model.eval()
        with torch.no_grad():
            val_recon = model(val_x)
            val_loss = criterion(val_recon, val_x).item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out / "model.pt")

        print(f"epoch={epoch:02d} train_mse={train_loss / len(train_x):.5f} val_mse={val_loss:.5f}")

    metrics = {"best_val_mse": best_val, "train_samples": len(train_x), "val_samples": len(val_x)}
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    train(epochs=args.epochs, output_dir=args.output)

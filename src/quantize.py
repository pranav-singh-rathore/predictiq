"""Quantise model and benchmark edge inference latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data_generator import GeneratorConfig, generate_dataset
from model import LSTMAutoencoder


def benchmark(model: torch.nn.Module, x: np.ndarray, runs: int = 200) -> float:
    model.eval()
    sample = torch.from_numpy(x[:1])
    with torch.no_grad():
        for _ in range(20):
            model(sample)
        start = time.perf_counter()
        for _ in range(runs):
            model(sample)
        elapsed_ms = (time.perf_counter() - start) * 1000 / runs
    return elapsed_ms


def quantize_and_benchmark(model_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    x, _ = generate_dataset(GeneratorConfig())

    model = LSTMAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    fp32_ms = benchmark(model, x)

    sample = torch.from_numpy(x[:1])
    scripted = torch.jit.trace(model, sample)
    scripted_path = output_dir / "model_quantised.pt"
    torch.jit.save(scripted, scripted_path)

    optimised_ms = benchmark(scripted, x)

    metrics = {
        "fp32_latency_ms": round(fp32_ms, 1),
        "optimised_latency_ms": round(optimised_ms, 1),
        "speedup_factor": round(fp32_ms / max(optimised_ms, 0.1), 1),
        "scripted_model": str(scripted_path),
        "note": "TorchScript optimised model; INT8 on-device via TFLite/ONNX on Raspberry Pi",
    }
    (output_dir / "latency_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    quantize_and_benchmark(args.model, args.output)

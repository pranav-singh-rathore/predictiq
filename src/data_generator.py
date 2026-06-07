"""Synthetic industrial sensor time-series with five fault patterns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FaultPattern(str, Enum):
    NORMAL = "normal"
    BEARING_WEAR = "bearing_wear"
    IMBALANCE = "imbalance"
    THERMAL_OVERLOAD = "thermal_overload"
    CAVITATION = "cavitation"
    MISALIGNMENT = "misalignment"


@dataclass(frozen=True)
class GeneratorConfig:
    seq_len: int = 64
    samples_per_class: int = 400
    seed: int = 42


def _base_signal(t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    vibration = 0.35 * np.sin(2 * np.pi * 30 * t) + 0.12 * np.sin(2 * np.pi * 90 * t)
    vibration += rng.normal(0, 0.02, size=t.shape)
    temperature = 42.0 + 0.8 * np.sin(2 * np.pi * 0.2 * t) + rng.normal(0, 0.15, size=t.shape)
    return vibration.astype(np.float32), temperature.astype(np.float32)


def _apply_fault(
    pattern: FaultPattern,
    vibration: np.ndarray,
    temperature: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    v, temp = vibration.copy(), temperature.copy()

    if pattern == FaultPattern.BEARING_WEAR:
        v += 0.35 * np.sin(2 * np.pi * 180 * t) + 0.12 * t
        temp += 1.2 * t
    elif pattern == FaultPattern.IMBALANCE:
        v += 0.42 * np.sin(2 * np.pi * 60 * t)
        temp += 0.35 * np.sin(2 * np.pi * 60 * t)
    elif pattern == FaultPattern.THERMAL_OVERLOAD:
        temp += 14.0 * (1 - np.exp(-3 * t)) + 0.6 * t
        v += 0.15 * t
    elif pattern == FaultPattern.CAVITATION:
        burst = (np.sin(2 * np.pi * 12 * t) > 0.85).astype(np.float32)
        v += burst * np.random.default_rng(1).normal(0, 1, size=v.shape).astype(np.float32) * 0.6
        temp += 1.2 * burst
    elif pattern == FaultPattern.MISALIGNMENT:
        v += 0.2 * np.sin(2 * np.pi * 120 * t) + 0.1 * np.sin(2 * np.pi * 240 * t)
        temp += 0.25 * t

    return v, temp


def generate_dataset(config: GeneratorConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    sequences: list[np.ndarray] = []
    labels: list[int] = []

    patterns = list(FaultPattern)
    for label_idx, pattern in enumerate(patterns):
        for _ in range(cfg.samples_per_class):
            t = np.linspace(0, 1, cfg.seq_len, dtype=np.float32)
            vib, temp = _base_signal(t, rng)
            if pattern != FaultPattern.NORMAL:
                vib, temp = _apply_fault(pattern, vib, temp, t)
            seq = np.stack([vib, temp], axis=1)
            sequences.append(seq)
            labels.append(label_idx)

    x = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    perm = rng.permutation(len(x))
    return x[perm], y[perm]

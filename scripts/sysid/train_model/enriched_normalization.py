# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-feature z-score normalization with JSON sidecar persistence."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def fit_stats(data: np.ndarray, eps: float = 1e-6) -> dict[str, np.ndarray]:
    """Compute per-feature mean/std; clip std to eps floor."""
    mean = data.mean(axis=0).astype(np.float32)
    std = data.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)
    return {"mean": mean, "std": std}


def apply_stats(data: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Apply z-score normalization using precomputed stats."""
    return (data - stats["mean"]) / stats["std"]


def save_stats(stats: dict[str, np.ndarray], path: str, metadata: dict[str, Any] | None = None) -> None:
    """Write stats and optional artifact-contract metadata to JSON."""
    payload = {"mean": stats["mean"].tolist(), "std": stats["std"].tolist()}
    if metadata is not None:
        payload["metadata"] = metadata
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_stats(path: str) -> dict[str, np.ndarray]:
    """Load stats back from JSON into float32 numpy arrays."""
    with open(path) as f:
        payload = json.load(f)
    return {
        "mean": np.array(payload["mean"], dtype=np.float32),
        "std": np.array(payload["std"], dtype=np.float32),
    }

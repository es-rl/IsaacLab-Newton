# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distributional metrics and SAGE trajectory loading for sim-to-real analysis."""

from __future__ import annotations

import ast
import csv
import math
import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MotionFeatures:
    """Joint features loaded from one SAGE motion directory.

    Attributes:
        joint_names: Joint names in column order.
        positions: Measured joint positions [rad], shape ``(sample_count, joint_count)``.
        velocities: Measured joint velocities [rad/s], shape ``(sample_count, joint_count)``.
        commands: Commanded joint positions [rad], shape ``(sample_count, joint_count)``, or ``None``.
    """

    joint_names: tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray
    commands: np.ndarray | None


@dataclass(frozen=True)
class FeatureMetrics:
    """Distributional metrics for one scalar joint feature.

    Attributes:
        wasserstein_raw: Wasserstein-1 distance in the feature's physical unit.
        wasserstein_normalized: Wasserstein-1 distance after real-reference normalization.
        mmd_rff_squared: Approximate squared RBF maximum mean discrepancy.
        bandwidth: RBF bandwidth in normalized feature units.
        real_mean: Mean of the real reference feature in its physical unit.
        real_scale: Standard deviation used to normalize the real reference feature.
        sim_samples: Number of simulated samples.
        real_samples: Number of real samples.
    """

    wasserstein_raw: float
    wasserstein_normalized: float
    mmd_rff_squared: float
    bandwidth: float
    real_mean: float
    real_scale: float
    sim_samples: int
    real_samples: int


def _validate_samples(samples: torch.Tensor, name: str) -> torch.Tensor:
    """Validate and flatten a scalar sample tensor."""
    samples = torch.as_tensor(samples)
    if samples.numel() == 0:
        raise ValueError(f"{name} must contain at least one sample")
    samples = samples.reshape(-1)
    if not torch.is_floating_point(samples):
        samples = samples.to(dtype=torch.float32)
    if not torch.isfinite(samples).all():
        raise ValueError(f"{name} contains non-finite samples")
    return samples


def wasserstein_distance_1d(sim_samples: torch.Tensor, real_samples: torch.Tensor) -> torch.Tensor:
    """Compute exact Wasserstein-1 distance between uniformly weighted 1-D samples."""
    sim_samples = _validate_samples(sim_samples, "sim_samples")
    real_samples = _validate_samples(real_samples, "real_samples").to(
        device=sim_samples.device, dtype=sim_samples.dtype
    )

    sim_sorted = torch.sort(sim_samples).values
    real_sorted = torch.sort(real_samples).values
    if sim_sorted.numel() == real_sorted.numel():
        return torch.mean(torch.abs(sim_sorted - real_sorted))

    combined = torch.sort(torch.cat((sim_sorted, real_sorted))).values
    deltas = combined[1:] - combined[:-1]
    evaluation_points = combined[:-1].contiguous()
    sim_cdf = torch.searchsorted(sim_sorted, evaluation_points, right=True) / sim_sorted.numel()
    real_cdf = torch.searchsorted(real_sorted, evaluation_points, right=True) / real_sorted.numel()
    return torch.sum(torch.abs(sim_cdf - real_cdf) * deltas)


def normalize_against_reference(
    samples: torch.Tensor, reference: torch.Tensor, epsilon: float = 1.0e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize samples and reference using the reference mean and standard deviation."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    reference = _validate_samples(reference, "reference")
    samples = torch.as_tensor(samples, device=reference.device, dtype=reference.dtype)
    if samples.numel() == 0:
        raise ValueError("samples must contain at least one sample")
    if not torch.isfinite(samples).all():
        raise ValueError("samples contains non-finite samples")
    mean = reference.mean()
    scale = reference.std(unbiased=False).clamp_min(epsilon)
    return (samples - mean) / scale, (reference - mean) / scale, mean, scale


def estimate_rbf_bandwidth(reference: torch.Tensor, max_samples: int = 2048, epsilon: float = 1.0e-6) -> torch.Tensor:
    """Estimate an RBF bandwidth with a deterministic median-distance heuristic."""
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    reference = _validate_samples(reference, "reference")
    if reference.numel() > max_samples:
        indices = torch.linspace(0, reference.numel() - 1, max_samples, device=reference.device).long()
        reference = reference[indices]
    if reference.numel() < 2:
        return torch.ones((), dtype=reference.dtype, device=reference.device)
    distances = torch.pdist(reference.unsqueeze(-1))
    positive = distances[distances > epsilon]
    if positive.numel() == 0:
        return torch.ones((), dtype=reference.dtype, device=reference.device)
    return positive.median().clamp_min(epsilon)


def _rff_parameters(
    bandwidth: torch.Tensor, num_features: int, seed: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    frequencies = torch.randn(num_features, generator=generator, device=device, dtype=dtype) / bandwidth
    phases = 2.0 * math.pi * torch.rand(num_features, generator=generator, device=device, dtype=dtype)
    return frequencies, phases


def _rff_mean(samples: torch.Tensor, frequencies: torch.Tensor, phases: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Compute a mean random-Fourier embedding without materializing all features."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    samples = torch.as_tensor(samples, device=frequencies.device, dtype=frequencies.dtype)
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)
    if samples.ndim != 2 or samples.shape[1] == 0:
        raise ValueError("samples must have shape (batch_size, sample_count) with at least one sample")
    if not torch.isfinite(samples).all():
        raise ValueError("samples contains non-finite samples")

    feature_sum = torch.zeros((samples.shape[0], frequencies.numel()), device=samples.device, dtype=samples.dtype)
    scale = math.sqrt(2.0 / frequencies.numel())
    for start in range(0, samples.shape[1], chunk_size):
        chunk = samples[:, start : start + chunk_size]
        embedding = torch.cos(chunk.unsqueeze(-1) * frequencies + phases) * scale
        feature_sum += embedding.sum(dim=1)
    return feature_sum / samples.shape[1]


def mmd_rff_squared(
    sim_samples: torch.Tensor,
    real_samples: torch.Tensor,
    *,
    bandwidth: torch.Tensor | float | None = None,
    num_features: int = 256,
    seed: int = 0,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute deterministic approximate squared RBF MMD for scalar samples.

    ``sim_samples`` may have shape ``(sample_count,)`` or
    ``(batch_size, sample_count)``. The result has one value per batch.
    """
    real_samples = _validate_samples(real_samples, "real_samples")
    sim_samples = torch.as_tensor(sim_samples, device=real_samples.device, dtype=real_samples.dtype)
    if sim_samples.ndim == 1:
        sim_samples = sim_samples.unsqueeze(0)
    if sim_samples.ndim != 2 or sim_samples.shape[1] == 0:
        raise ValueError("sim_samples must have shape (batch_size, sample_count) with at least one sample")
    if not torch.isfinite(sim_samples).all():
        raise ValueError("sim_samples contains non-finite samples")

    if bandwidth is None:
        bandwidth_tensor = estimate_rbf_bandwidth(real_samples)
    else:
        bandwidth_tensor = torch.as_tensor(bandwidth, device=real_samples.device, dtype=real_samples.dtype)
        if bandwidth_tensor.ndim != 0 or not torch.isfinite(bandwidth_tensor) or bandwidth_tensor <= 0:
            raise ValueError("bandwidth must be a finite positive scalar")

    frequencies, phases = _rff_parameters(bandwidth_tensor, num_features, seed, real_samples.device, real_samples.dtype)
    sim_mean = _rff_mean(sim_samples, frequencies, phases, chunk_size)
    real_mean = _rff_mean(real_samples.unsqueeze(0), frequencies, phases, chunk_size)
    return torch.sum((sim_mean - real_mean) ** 2, dim=1).clamp_min(0.0), bandwidth_tensor


def compute_feature_metrics(
    sim_samples: torch.Tensor | np.ndarray,
    real_samples: torch.Tensor | np.ndarray,
    *,
    num_features: int = 256,
    seed: int = 0,
    chunk_size: int = 1024,
    epsilon: float = 1.0e-6,
) -> FeatureMetrics:
    """Compute raw/normalized Wasserstein distance and normalized RBF-MMD²."""
    sim_tensor = _validate_samples(torch.as_tensor(sim_samples, dtype=torch.float64), "sim_samples")
    real_tensor = _validate_samples(torch.as_tensor(real_samples, dtype=torch.float64), "real_samples")
    sim_normalized, real_normalized, real_mean, real_scale = normalize_against_reference(
        sim_tensor, real_tensor, epsilon
    )
    mmd, bandwidth = mmd_rff_squared(
        sim_normalized,
        real_normalized,
        num_features=num_features,
        seed=seed,
        chunk_size=chunk_size,
    )
    return FeatureMetrics(
        wasserstein_raw=float(wasserstein_distance_1d(sim_tensor, real_tensor)),
        wasserstein_normalized=float(wasserstein_distance_1d(sim_normalized, real_normalized)),
        mmd_rff_squared=float(mmd[0]),
        bandwidth=float(bandwidth),
        real_mean=float(real_mean),
        real_scale=float(real_scale),
        sim_samples=sim_tensor.numel(),
        real_samples=real_tensor.numel(),
    )


def compute_batched_distributional_score(
    sim_positions: torch.Tensor,
    real_positions: torch.Tensor,
    sim_velocities: torch.Tensor,
    real_velocities: torch.Tensor,
    *,
    objective: str,
    num_features: int = 256,
    seed: int = 0,
    chunk_size: int = 1024,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Score batched SysID trajectories with equally weighted joint features.

    Args:
        sim_positions: Simulated joint positions [rad], shape ``(environment_count, sample_count, joint_count)``.
        real_positions: Real joint positions [rad], shape ``(sample_count, joint_count)``.
        sim_velocities: Simulated joint velocities [rad/s], same shape as ``sim_positions``.
        real_velocities: Real joint velocities [rad/s], same shape as ``real_positions``.
        objective: Distributional objective, either ``"wasserstein"`` or ``"mmd"``.
        num_features: Number of random Fourier features used for MMD.
        seed: Random Fourier feature seed.
        chunk_size: Maximum trajectory samples embedded in one MMD operation.
        epsilon: Minimum real-reference normalization scale.

    Returns:
        One dimensionless score per simulated environment.
    """
    if objective not in ("wasserstein", "mmd"):
        raise ValueError(f"Unsupported distributional objective: {objective}")
    if sim_positions.ndim != 3 or real_positions.ndim != 2:
        raise ValueError("position tensors must have shapes (E, T, J) and (T, J)")
    if sim_velocities.shape != sim_positions.shape or real_velocities.shape != real_positions.shape:
        raise ValueError("velocity tensor shapes must match their corresponding position tensors")
    if sim_positions.shape[1:] != real_positions.shape:
        raise ValueError("sim and real trajectories must have matching sample and joint dimensions")
    for name, tensor in (
        ("sim_positions", sim_positions),
        ("real_positions", real_positions),
        ("sim_velocities", sim_velocities),
        ("real_velocities", real_velocities),
    ):
        if tensor.numel() == 0 or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain finite samples")

    scores = torch.zeros(sim_positions.shape[0], device=sim_positions.device, dtype=sim_positions.dtype)
    feature_count = 2 * sim_positions.shape[2]
    for feature_index, (sim_values, real_values) in enumerate(
        ((sim_positions, real_positions), (sim_velocities, real_velocities))
    ):
        for joint_index in range(sim_positions.shape[2]):
            sim_feature = sim_values[:, :, joint_index]
            real_feature = real_values[:, joint_index].to(device=sim_feature.device, dtype=sim_feature.dtype)
            sim_normalized, real_normalized, _, _ = normalize_against_reference(sim_feature, real_feature, epsilon)
            if objective == "wasserstein":
                sim_sorted = torch.sort(sim_normalized, dim=1).values
                real_sorted = torch.sort(real_normalized).values.unsqueeze(0)
                scores += torch.mean(torch.abs(sim_sorted - real_sorted), dim=1)
            else:
                mmd, _ = mmd_rff_squared(
                    sim_normalized,
                    real_normalized,
                    num_features=num_features,
                    seed=seed + feature_index * sim_positions.shape[2] + joint_index,
                    chunk_size=chunk_size,
                )
                scores += mmd
    return scores / feature_count


def _read_joint_names(motion_dir: str) -> tuple[str, ...]:
    path = os.path.join(motion_dir, "joint_list.txt")
    with open(path) as file:
        joint_names = tuple(line.strip() for line in file if line.strip())
    if not joint_names:
        raise ValueError(f"No joint names found in {path}")
    return joint_names


def _read_vector_csv(path: str, vector_fields: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    timestamps: list[float] = []
    vectors: dict[str, list[list[float]]] = {field: [] for field in vector_fields}
    with open(path) as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "timestamp" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a timestamp column")
        missing = [field for field in vector_fields if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            for field in vector_fields:
                value = ast.literal_eval(row[field])
                if not isinstance(value, (list, tuple)):
                    raise ValueError(f"{field} in {path} must contain a list")
                vectors[field].append([float(item) for item in value])
    if not timestamps:
        raise ValueError(f"{path} contains no samples")
    return np.asarray(timestamps, dtype=np.float64), {
        field: np.asarray(values, dtype=np.float64) for field, values in vectors.items()
    }


def _event_window_seconds(motion_dir: str, timestamps: np.ndarray) -> tuple[float, float]:
    scale = 1.0e-6 if float(np.nanmax(np.abs(timestamps))) > 1000.0 else 1.0
    start = float(timestamps[0]) * scale
    end = float(timestamps[-1]) * scale
    event_path = os.path.join(motion_dir, "event.csv")
    if not os.path.isfile(event_path):
        return start, end

    with open(event_path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            event = row.get("event", "").upper()
            timestamp = float(row["timestamp"]) * scale
            if event == "MOTION_START":
                start = timestamp
            elif event == "DISABLE":
                end = timestamp
    if end < start:
        raise ValueError(f"Invalid event window in {event_path}: DISABLE precedes MOTION_START")
    return start, end


def _crop_to_event_window(
    motion_dir: str, timestamps: np.ndarray, values: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    scale = 1.0e-6 if float(np.nanmax(np.abs(timestamps))) > 1000.0 else 1.0
    timestamps_seconds = timestamps * scale
    start, end = _event_window_seconds(motion_dir, timestamps)
    mask = (timestamps_seconds >= start) & (timestamps_seconds <= end)
    if not np.any(mask):
        raise ValueError(f"No samples in event window for {motion_dir}")
    return {name: data[mask] for name, data in values.items()}


def load_sage_motion_features(motion_dir: str) -> MotionFeatures:
    """Load position, velocity, and optional command distributions from a SAGE motion."""
    joint_names = _read_joint_names(motion_dir)
    state_path = os.path.join(motion_dir, "state_motor.csv")
    state_times, state_values = _read_vector_csv(state_path, ("positions", "velocities"))
    state_values = _crop_to_event_window(motion_dir, state_times, state_values)
    for name, values in state_values.items():
        if values.ndim != 2 or values.shape[1] != len(joint_names):
            raise ValueError(
                f"{name} in {state_path} has {values.shape[1] if values.ndim == 2 else 'invalid'} "
                f"joints; expected {len(joint_names)}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} in {state_path} contains non-finite samples")

    commands = None
    control_path = os.path.join(motion_dir, "control.csv")
    if os.path.isfile(control_path):
        control_times, control_values = _read_vector_csv(control_path, ("positions",))
        control_values = _crop_to_event_window(motion_dir, control_times, control_values)
        commands = control_values["positions"]
        if commands.ndim != 2 or commands.shape[1] != len(joint_names):
            raise ValueError(
                f"positions in {control_path} has {commands.shape[1] if commands.ndim == 2 else 'invalid'} "
                f"joints; expected {len(joint_names)}"
            )
        if not np.isfinite(commands).all():
            raise ValueError(f"positions in {control_path} contains non-finite samples")

    return MotionFeatures(
        joint_names=joint_names,
        positions=state_values["positions"],
        velocities=state_values["velocities"],
        commands=commands,
    )


def load_sage_joint_velocities(motion_dir: str, joint_names: list[str]) -> np.ndarray:
    """Load uncropped measured joint velocities [rad/s] in requested column order."""
    all_joint_names = _read_joint_names(motion_dir)
    state_path = os.path.join(motion_dir, "state_motor.csv")
    _, state_values = _read_vector_csv(state_path, ("velocities",))
    joint_indices = {name: index for index, name in enumerate(all_joint_names)}
    missing = [name for name in joint_names if name not in joint_indices]
    if missing:
        raise ValueError(f"{state_path} is missing requested joints: {missing}")
    velocities = state_values["velocities"][:, [joint_indices[name] for name in joint_names]]
    if velocities.ndim != 2 or not np.isfinite(velocities).all():
        raise ValueError(f"velocities in {state_path} must be a finite two-dimensional array")
    return velocities

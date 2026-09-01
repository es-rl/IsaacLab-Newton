# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared training and deployment contract for enriched actuator GRUs."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

H1_DEPLOYABLE_FEATURE_LAYOUT = "deployable20"
H1_LEGACY_FEATURE_LAYOUT = "legacy24"
H1_DEPLOYABLE_FEATURE_BLOCKS = (
    "position",
    "position_error",
    "velocity",
    "pd_hint",
    "previous_torque",
)
H1_LEGACY_FEATURE_BLOCKS = (
    "position",
    "position_error",
    "velocity",
    "pd_hint",
    "qfrc_bias",
    "previous_torque",
)
H1_JOINT_NAMES = (
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
)
H1_NUM_JOINTS = len(H1_JOINT_NAMES)
H1_DEPLOYABLE_INPUT_SIZE = len(H1_DEPLOYABLE_FEATURE_BLOCKS) * H1_NUM_JOINTS
H1_LEGACY_INPUT_SIZE = len(H1_LEGACY_FEATURE_BLOCKS) * H1_NUM_JOINTS
H1_RECOMMENDED_RESIDUAL_SCALE = 0.33


@dataclass(frozen=True)
class EnrichedGruContract:
    """Dimensions inferred from an exported :class:`ForceResidualGRU`."""

    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int


def infer_enriched_gru_contract(model: torch.nn.Module) -> EnrichedGruContract:
    """Infer recurrent dimensions from a scripted or eager enriched GRU."""
    parameters = dict(model.named_parameters())
    required = ("gru.weight_ih_l0", "gru.weight_hh_l0", "force_head.weight")
    missing = [name for name in required if name not in parameters]
    if missing:
        raise ValueError(f"Enriched GRU is missing parameters: {missing}")

    weight_ih = parameters["gru.weight_ih_l0"]
    weight_hh = parameters["gru.weight_hh_l0"]
    force_head = parameters["force_head.weight"]
    hidden_size = int(weight_hh.shape[1])
    if int(weight_ih.shape[0]) != 3 * hidden_size:
        raise ValueError("Enriched actuator checkpoint is not a GRU")

    layer_indices = sorted(
        int(name.removeprefix("gru.weight_ih_l"))
        for name in parameters
        if name.startswith("gru.weight_ih_l") and name.removeprefix("gru.weight_ih_l").isdigit()
    )
    if layer_indices != list(range(len(layer_indices))):
        raise ValueError(f"GRU layers are not contiguous: {layer_indices}")

    return EnrichedGruContract(
        input_size=int(weight_ih.shape[1]),
        hidden_size=hidden_size,
        num_layers=len(layer_indices),
        output_size=int(force_head.shape[0]),
    )


def validate_normalization_stats(stats: dict[str, Any], expected_input_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and return finite per-feature normalization tensors."""
    if "mean" not in stats or "std" not in stats:
        raise ValueError("Normalization stats must contain mean and std")
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    std = torch.as_tensor(stats["std"], dtype=torch.float32)
    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError("Normalization mean and std must be one-dimensional")
    if mean.numel() != expected_input_size or std.numel() != expected_input_size:
        raise ValueError(
            f"Normalization size mismatch: expected {expected_input_size}, "
            f"got mean={mean.numel()} and std={std.numel()}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Normalization stats must be finite")
    if not torch.all(std > 0):
        raise ValueError("Normalization standard deviations must be positive")
    return mean, std


def build_h1_deployable_features(
    position: torch.Tensor,
    position_error: torch.Tensor,
    velocity: torch.Tensor,
    pd_hint: torch.Tensor,
    previous_torque: torch.Tensor,
) -> torch.Tensor:
    """Build the H1 deployable 20-feature tensor in canonical block order."""
    blocks = (position, position_error, velocity, pd_hint, previous_torque)
    if any(block.shape != position.shape for block in blocks):
        raise ValueError(f"H1 feature blocks must share one shape, got {[tuple(block.shape) for block in blocks]}")
    if position.ndim != 2 or position.shape[1] != H1_NUM_JOINTS:
        raise ValueError(f"H1 feature blocks must have shape [batch, {H1_NUM_JOINTS}]")
    return torch.cat(blocks, dim=-1)


def make_h1_deployable_metadata(
    *,
    hidden_size: int,
    num_layers: int,
    force_bound: float,
    kp: float,
    kd: float,
) -> dict[str, Any]:
    """Create the versioned artifact metadata for H1 enriched residuals."""
    return {
        "schema_version": 1,
        "model_type": "enriched_residual",
        "robot": "h1",
        "joint_names": list(H1_JOINT_NAMES),
        "feature_layout": H1_DEPLOYABLE_FEATURE_LAYOUT,
        "feature_blocks": list(H1_DEPLOYABLE_FEATURE_BLOCKS),
        "input_size": H1_DEPLOYABLE_INPUT_SIZE,
        "output_size": H1_NUM_JOINTS,
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "force_bound": float(force_bound),
        "kp_factory": float(kp),
        "kd_factory": float(kd),
        "target": "tau_real_minus_pd",
        "recommended_residual_scale": H1_RECOMMENDED_RESIDUAL_SCALE,
    }


def validate_h1_deployable_metadata(
    stats: dict[str, Any], contract: EnrichedGruContract, residual_scale: float
) -> tuple[float, float]:
    """Validate that H1 artifacts and runtime settings share one contract."""
    metadata = stats.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("H1 enriched residual stats must contain metadata")

    expected = {
        "schema_version": 1,
        "model_type": "enriched_residual",
        "robot": "h1",
        "joint_names": list(H1_JOINT_NAMES),
        "feature_layout": H1_DEPLOYABLE_FEATURE_LAYOUT,
        "feature_blocks": list(H1_DEPLOYABLE_FEATURE_BLOCKS),
        "input_size": H1_DEPLOYABLE_INPUT_SIZE,
        "output_size": H1_NUM_JOINTS,
        "target": "tau_real_minus_pd",
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise ValueError(f"H1 enriched residual metadata mismatch: {mismatches}")

    dimensions = {
        "input_size": contract.input_size,
        "output_size": contract.output_size,
        "hidden_size": contract.hidden_size,
        "num_layers": contract.num_layers,
    }
    dimension_mismatches = {
        key: (metadata.get(key), value) for key, value in dimensions.items() if metadata.get(key) != value
    }
    if dimension_mismatches:
        raise ValueError(f"H1 checkpoint dimensions do not match metadata: {dimension_mismatches}")
    if not math.isfinite(residual_scale) or residual_scale <= 0:
        raise ValueError("residual_scale must be finite and positive")

    kp = float(metadata.get("kp_factory", float("nan")))
    kd = float(metadata.get("kd_factory", float("nan")))
    if not math.isfinite(kp) or kp < 0 or not math.isfinite(kd) or kd < 0:
        raise ValueError("H1 metadata must contain finite, non-negative kp_factory and kd_factory")
    return kp, kd


class EnrichedResidualRuntime:
    """Stateful four-joint H1 residual inference independent of Isaac Lab APIs."""

    def __init__(
        self,
        model: torch.nn.Module,
        stats: dict[str, Any],
        residual_scale: float,
        num_envs: int,
        device: torch.device | str,
    ) -> None:
        self.model = model
        self.contract = infer_enriched_gru_contract(model)
        self.kp, self.kd = validate_h1_deployable_metadata(stats, self.contract, residual_scale)
        mean, std = validate_normalization_stats(stats, self.contract.input_size)
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.residual_scale = float(residual_scale)
        self.hidden = torch.zeros(
            self.contract.num_layers,
            num_envs,
            self.contract.hidden_size,
            dtype=torch.float32,
            device=device,
        )
        self.previous_torque = torch.zeros(
            num_envs,
            self.contract.output_size,
            dtype=torch.float32,
            device=device,
        )

    def reset(self) -> None:
        """Clear recurrent and autoregressive state at a rollout boundary."""
        self.hidden.zero_()
        self.previous_torque.zero_()

    def step(
        self,
        position: torch.Tensor,
        position_target: torch.Tensor,
        velocity: torch.Tensor,
        stiffness: torch.Tensor,
        damping: torch.Tensor,
        existing_effort: torch.Tensor,
        clip_effort: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict scaled residual effort and update previous total torque."""
        position_error = position_target - position
        expected_stiffness = torch.full_like(stiffness, self.kp)
        expected_damping = torch.full_like(damping, self.kd)
        if not torch.allclose(stiffness, expected_stiffness, rtol=0.0, atol=1e-5):
            raise ValueError(f"Runtime stiffness must match training kp={self.kp}")
        if not torch.allclose(damping, expected_damping, rtol=0.0, atol=1e-5):
            raise ValueError(f"Runtime damping must match training kd={self.kd}")

        pd_hint = stiffness * position_error - damping * velocity
        features = build_h1_deployable_features(
            position,
            position_error,
            velocity,
            pd_hint,
            self.previous_torque,
        )
        normalized = (features - self.mean) / self.std
        with torch.inference_mode():
            residual_sequence, hidden = self.model(normalized.unsqueeze(1), self.hidden)
        self.hidden.copy_(hidden)
        scaled_residual = residual_sequence[:, -1, :].to(torch.float32) * self.residual_scale
        feedforward_effort = existing_effort + scaled_residual
        computed_total = pd_hint + feedforward_effort
        applied_total = clip_effort(computed_total)
        self.previous_torque.copy_(applied_total.detach())
        return feedforward_effort, computed_total, applied_total

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared enriched GRU training and deployment contract."""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

_TRAIN_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_model")
sys.path.insert(0, _TRAIN_MODEL_DIR)

from enriched_contract import (  # noqa: E402
    H1_DEPLOYABLE_INPUT_SIZE,
    EnrichedResidualRuntime,
    build_h1_deployable_features,
    infer_enriched_gru_contract,
    make_h1_deployable_metadata,
    validate_h1_deployable_metadata,
    validate_normalization_stats,
)
from enriched_model import ForceResidualGRU  # noqa: E402


def _model(input_size: int = 20, hidden_size: int = 128, num_layers: int = 2) -> ForceResidualGRU:
    return ForceResidualGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_joints=4,
        force_bound=2.0,
        dropout=0.0,
    )


def _stats(model: ForceResidualGRU) -> dict:
    contract = infer_enriched_gru_contract(model)
    return {
        "mean": [0.0] * contract.input_size,
        "std": [1.0] * contract.input_size,
        "metadata": make_h1_deployable_metadata(
            hidden_size=contract.hidden_size,
            num_layers=contract.num_layers,
            force_bound=2.0,
            kp=60.0,
            kd=1.5,
        ),
    }


def test_infers_contract_from_torchscript_artifact() -> None:
    scripted = torch.jit.script(_model(input_size=20, hidden_size=16, num_layers=1))
    contract = infer_enriched_gru_contract(scripted)
    assert contract.input_size == 20
    assert contract.hidden_size == 16
    assert contract.num_layers == 1
    assert contract.output_size == 4


def test_infers_g1_hidden_size_from_checkpoint_parameters() -> None:
    contract = infer_enriched_gru_contract(_model(input_size=24, hidden_size=192, num_layers=2))
    assert contract.input_size == 24
    assert contract.hidden_size == 192
    assert contract.num_layers == 2
    assert contract.output_size == 4


@pytest.mark.parametrize(
    ("mean", "std", "match"),
    [
        ([0.0] * 19, [1.0] * 20, "size mismatch"),
        ([0.0] * 20, [0.0] * 20, "must be positive"),
        ([0.0] * 19 + [float("nan")], [1.0] * 20, "must be finite"),
    ],
)
def test_normalization_validation_rejects_invalid_stats(mean, std, match) -> None:
    with pytest.raises(ValueError, match=match):
        validate_normalization_stats({"mean": mean, "std": std}, H1_DEPLOYABLE_INPUT_SIZE)


def test_builds_h1_features_in_exact_block_order() -> None:
    blocks = [torch.full((2, 4), float(index)) for index in range(1, 6)]
    features = build_h1_deployable_features(*blocks)
    assert features.shape == (2, 20)
    for index, block in enumerate(blocks):
        torch.testing.assert_close(features[:, index * 4 : (index + 1) * 4], block)


def test_metadata_must_match_checkpoint_dimensions() -> None:
    model = _model(hidden_size=128)
    contract = infer_enriched_gru_contract(model)
    stats = _stats(model)
    stats["metadata"]["hidden_size"] = 192
    with pytest.raises(ValueError, match="checkpoint dimensions"):
        validate_h1_deployable_metadata(stats, contract, residual_scale=0.33)


def test_runtime_scales_residual_and_feeds_back_clipped_total_torque() -> None:
    model = _model(hidden_size=8, num_layers=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.force_head.bias.fill_(math.atanh(0.5))

    runtime = EnrichedResidualRuntime(
        model=model,
        stats=_stats(model),
        residual_scale=0.33,
        num_envs=1,
        device="cpu",
    )
    position = torch.zeros(1, 4)
    position_target = torch.full((1, 4), 0.1)
    velocity = torch.zeros(1, 4)
    stiffness = torch.full((1, 4), 60.0)
    damping = torch.full((1, 4), 1.5)
    existing_effort = torch.full((1, 4), 0.2)

    feedforward, computed, applied = runtime.step(
        position,
        position_target,
        velocity,
        stiffness,
        damping,
        existing_effort,
        lambda torque: torque.clamp(-5.0, 5.0),
    )

    torch.testing.assert_close(feedforward, torch.full((1, 4), 0.53))
    torch.testing.assert_close(computed, torch.full((1, 4), 6.53))
    torch.testing.assert_close(applied, torch.full((1, 4), 5.0))
    torch.testing.assert_close(runtime.previous_torque, applied)

    runtime.reset()
    assert torch.count_nonzero(runtime.hidden) == 0
    assert torch.count_nonzero(runtime.previous_torque) == 0


def test_runtime_rejects_pd_gains_that_differ_from_training() -> None:
    model = _model(hidden_size=8, num_layers=1)
    runtime = EnrichedResidualRuntime(model, _stats(model), 0.33, 1, "cpu")
    values = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="stiffness"):
        runtime.step(
            values,
            values,
            values,
            torch.full((1, 4), 40.0),
            torch.full((1, 4), 1.5),
            values,
            lambda torque: torque,
        )

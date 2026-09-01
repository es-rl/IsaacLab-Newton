# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Text-level regression checks for enriched GRU benchmark dispatch.

The benchmark module imports the Isaac Sim runtime at module scope, so these
checks intentionally validate its wiring without launching the simulator.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INPUT_DIR = _REPO_ROOT / "input"
sys.path.insert(0, str(_INPUT_DIR))

from run_configs import load_run_cfg  # noqa: E402

_BENCHMARK = _REPO_ROOT / "scripts" / "sim2real_gap" / "newton_benchmark.py"
_ENTRYPOINT = _REPO_ROOT / "scripts" / "sim2real_gap" / "run_benchmark.py"
_H1_TRAINER = _REPO_ROOT / "scripts" / "sysid" / "train_model" / "train_gru_enriched_h1.py"
_H1_RUN_CONFIG = _REPO_ROOT / "input" / "run_configs" / "h1" / "h1_enriched_residual.yaml"


def test_g1_enriched_hidden_state_is_inferred_from_checkpoint() -> None:
    source = _BENCHMARK.read_text()
    assert "hidden = torch.zeros(2, num_envs, 128" not in source
    assert "contract = infer_enriched_gru_contract(model)" in source
    assert "contract.hidden_size" in source


def test_h1_enriched_residual_has_dedicated_coupled_dispatch() -> None:
    source = _BENCHMARK.read_text()
    assert 'elif model_type == "enriched_residual":' in source
    assert 'if act_cfg.get("model_type") == "enriched_residual":' in source
    assert "_patch_h1_enriched_residual(" in source
    assert "residual_scale" in source


def test_explicit_h1_run_config_is_loadable_and_wired_to_entrypoint() -> None:
    config = load_run_cfg("h1", str(_H1_RUN_CONFIG))
    assert config["actuator"]["model_type"] == "enriched_residual"
    assert config["actuator"]["residual_scale"] == 0.33

    source = _ENTRYPOINT.read_text()
    assert '"--run-config"' in source
    assert "load_run_cfg(args.robot_name, args.run_config)" in source


def test_h1_run_config_selects_matching_artifacts_and_scale() -> None:
    source = _H1_RUN_CONFIG.read_text()
    assert "model_type: enriched_residual" in source
    assert "h1_arm_enriched_residual_deployable20_script.pt" in source
    assert "h1_arm_enriched_residual_deployable20_stats.json" in source
    assert "residual_scale: 0.33" in source


def test_h1_trainer_defaults_to_deployable20_contract() -> None:
    source = _H1_TRAINER.read_text()
    assert "default=H1_DEPLOYABLE_FEATURE_LAYOUT" in source
    assert 'default=os.path.join(SAVE_DIR, "h1_arm_enriched_residual_deployable20.pt")' in source
    assert "make_h1_deployable_metadata(" in source

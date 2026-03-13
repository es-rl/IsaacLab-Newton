"""Newton-backend joint motion benchmark, compatible with SAGE CSV output format.

This module provides a Newton physics adaptation of SAGE's JointMotionBenchmark.
It uses Isaac Lab's InteractiveScene + SimulationContext with Newton solver
configuration (the same pattern as ManagerBasedRLEnv), producing identical CSV
output so that sage.analysis tools work unchanged.
"""

import csv
import json
import os
import sys
from collections import deque

import numpy as np
import torch
import warp as wp
from scipy.interpolate import interp1d

import isaaclab.sim as sim_utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "input"))
from actuator_models import load_actuator_params, load_implicit_actuator_cfg
from isaaclab.actuators import ActuatorNetLSTMCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import H1_MINIMAL_CFG

from spawn_utils import spawn_from_usd_with_fixed_base

from actuator_models import load_dc_motor_cfg, load_fmu_actuator_cfg

# Local USD assets (avoids dependency on cloud-hosted Omniverse Nucleus server)
_H1_LOCAL_USD = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../input/robot_models/h1_minimal/h1_minimal.usda")
)
_UR10_LOCAL_USD = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../input/robot_models/ur10/ur10/ur10.usd")
)

# Reuse SAGE's pure-Python utility functions
from sage.simulation import get_motion_files, get_motion_name, log_message  # noqa: F401

# ---------------------------------------------------------------------------
# Auto-convert regular PyTorch checkpoints to TorchScript
# ---------------------------------------------------------------------------
# Isaac Lab loads LSTM actuator nets via torch.jit.load, which requires a
# TorchScript archive (.pt with code/ and constants.pkl).  If the user points
# network_file at a plain torch.save() checkpoint (state_dict only), we detect
# the mismatch, rebuild the model, trace it, and cache the scripted version
# next to the original file so the conversion only happens once.


def _ensure_torchscript(pt_path: str) -> str:
    """Return a TorchScript .pt path, auto-converting a checkpoint if needed.

    If *pt_path* is already a valid TorchScript archive it is returned as-is.
    Otherwise the file is assumed to be a ``torch.save(state_dict)`` checkpoint
    for a TorqueLSTM or TorqueGRU model.  The model is reconstructed from the
    weight shapes, traced, and saved to ``<name>_scripted.pt`` alongside the
    original.

    Supports both LSTM and GRU architectures (auto-detected from state_dict
    keys).  GRU models are wrapped to match Isaac Lab's LSTM interface:
    ``forward(input, (hidden, cell)) -> (output, (hidden, cell))``, where
    ``cell`` is a pass-through dummy (GRU has no cell state).
    """
    import zipfile

    if not os.path.isfile(pt_path):
        return pt_path  # Let downstream report the missing-file error

    # Quick probe: TorchScript archives contain a constants.pkl entry
    try:
        with zipfile.ZipFile(pt_path) as zf:
            names = zf.namelist()
            if any(n.endswith("constants.pkl") for n in names):
                return pt_path  # Already TorchScript
    except zipfile.BadZipFile:
        return pt_path  # Not a zip at all — let downstream error

    # Derive scripted path: foo.pt -> foo_scripted.pt
    base, ext = os.path.splitext(pt_path)
    scripted_path = f"{base}_scripted{ext}"

    if os.path.isfile(scripted_path):
        log_message(f"Using cached TorchScript model: {os.path.basename(scripted_path)}")
        return scripted_path

    # --- Reconstruct model and wrap for Isaac Lab's interface ---
    #
    # Isaac Lab's ActuatorNetLSTM calls:
    #   output, (hidden, cell) = network(input, (hidden, cell))
    #
    # We wrap both LSTM and GRU to match this signature.
    log_message(f"Converting checkpoint to TorchScript: {os.path.basename(pt_path)}")
    import torch.nn as nn

    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)

    # Detect architecture from state_dict keys
    has_lstm = any(k.startswith("lstm.") for k in state_dict)
    has_gru = any(k.startswith("gru.") for k in state_dict)

    if has_lstm:
        rnn_prefix = "lstm"
        gate_factor = 4  # LSTM: 4 gates (i, f, g, o)
    elif has_gru:
        rnn_prefix = "gru"
        gate_factor = 3  # GRU: 3 gates (r, z, n)
    else:
        raise ValueError(f"Cannot detect RNN type in {pt_path}. Keys: {list(state_dict.keys())[:10]}")

    # Infer architecture from weight shapes
    # weight_ih_l0: (gate_factor * hidden_dim, input_size)
    weight_key = f"{rnn_prefix}.weight_ih_l0"
    input_size = state_dict[weight_key].shape[1]
    hidden_dim = state_dict[weight_key].shape[0] // gate_factor
    num_layers = max(
        int(k.split("_l")[1].split(".")[0])
        for k in state_dict
        if k.startswith(f"{rnn_prefix}.") and "_l" in k
    ) + 1

    if has_lstm:
        class _IsaacLabRNNWrapper(nn.Module):
            """TorqueLSTM wrapped to match Isaac Lab's ActuatorNetLSTM interface."""

            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size, hidden_size=hidden_dim,
                    num_layers=num_layers, batch_first=True,
                )
                self.head = nn.Linear(hidden_dim, 1)

            def forward(
                self, x: torch.Tensor, hx: tuple[torch.Tensor, torch.Tensor],
            ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                out, (h_new, c_new) = self.lstm(x, hx)
                torque = self.head(out[:, -1, :])
                return torque, (h_new, c_new)
    else:
        class _IsaacLabRNNWrapper(nn.Module):
            """TorqueGRU wrapped to match Isaac Lab's ActuatorNetLSTM interface.

            The GRU is registered as ``self.lstm`` so that Newton's
            ``ActuatorNetLSTM.__init__`` (which hardcodes ``self.network.lstm``)
            can find it.  Both LSTM and GRU have 4 state_dict entries per layer,
            so Newton's ``len(state_dict()) // 4`` correctly infers num_layers.

            GRU has no cell state — the cell tensor is accepted and passed
            through unchanged as a dummy.
            """

            def __init__(self):
                super().__init__()
                # Named 'lstm' so Newton's ActuatorNetLSTM.__init__ resolves
                # self.network.lstm correctly
                self.lstm = nn.GRU(
                    input_size=input_size, hidden_size=hidden_dim,
                    num_layers=num_layers, batch_first=True,
                )
                self.head = nn.Linear(hidden_dim, 1)

            def forward(
                self, x: torch.Tensor, hx: tuple[torch.Tensor, torch.Tensor],
            ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                h, c_passthrough = hx
                out, h_new = self.lstm(x, h)
                torque = self.head(out[:, -1, :])
                return torque, (h_new, c_passthrough)

    model = _IsaacLabRNNWrapper()
    # For GRU: remap state_dict keys from gru.* to lstm.* to match wrapper attribute name
    if has_gru:
        state_dict = {k.replace("gru.", "lstm."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Trace with dummy inputs matching Isaac Lab's call signature
    dummy_x = torch.randn(1, 1, input_size)
    dummy_h = torch.zeros(num_layers, 1, hidden_dim)
    dummy_c = torch.zeros(num_layers, 1, hidden_dim)
    traced = torch.jit.trace(model, (dummy_x, (dummy_h, dummy_c)))
    torch.jit.save(traced, scripted_path)
    rnn_type = "LSTM" if has_lstm else "GRU"
    log_message(
        f"Saved TorchScript model: {os.path.basename(scripted_path)} "
        f"({rnn_type}, input={input_size}, hidden={hidden_dim}, layers={num_layers})"
    )
    return scripted_path


# ---------------------------------------------------------------------------
# Newton ActuatorNetLSTM bug fix + optional normalization
# ---------------------------------------------------------------------------
# Newton's ActuatorNetLSTM.compute() has a joint indexing bug in multi-actuator
# setups: it reads full-robot data arrays (19 joints) but sea_input is sized
# for the actuator group only (8 arm joints). Updated compute() on the
# instance to fix the indexing. For the 3-input sysid model, also applies
# percentile clipping + z-normalization to match training preprocessing.

# Normalization stats for the sysid LSTM (3-input model: position, position_error, velocity).
# Pre-computed from the type 3 experiment dataset (12.24M samples).
_SYSID_STATS = {
    "position":       {"mean":  0.7976, "std": 0.2867, "p1": -0.1217, "p99": 1.7152},
    "position_error": {"mean": -0.0122, "std": 0.1438, "p1": -0.5197, "p99": 0.3289},
    "velocity":       {"mean": -0.0166, "std": 2.5782, "p1": -7.0728, "p99": 7.0962},
    "torque":         {"mean": -0.0091, "std": 3.8534, "p1": -12.9865, "p99": 12.9973},
}


def _load_sysid_stats(network_file: str) -> dict:
    """Load normalization stats from a sidecar JSON file next to the model.

    Looks for ``<model_name>_stats.json`` alongside the ``.pt`` file.
    The JSON has joint-prefixed keys under ``normalization`` (e.g.
    ``elbow_position``, ``pitch_velocity``).  This function strips the
    prefix and returns canonical keys (``position``, ``position_error``,
    ``velocity``, ``torque``) matching ``_SYSID_STATS`` format.

    Falls back to the global ``_SYSID_STATS`` if no sidecar exists.
    """
    # Also try stripping the _scripted suffix that _ensure_torchscript adds
    base, _ = os.path.splitext(network_file)
    candidates = [base + "_stats.json"]
    if base.endswith("_scripted"):
        candidates.append(base.removesuffix("_scripted") + "_stats.json")

    stats_path = None
    for path in candidates:
        if os.path.isfile(path):
            stats_path = path
            break

    if stats_path is None:
        log_message(
            f"  Stats: using global fallback (_SYSID_STATS) — "
            f"no sidecar found at {candidates[0]}"
        )
        _log_norm_stats(_SYSID_STATS, "global fallback")
        return _SYSID_STATS

    log_message(f"  Stats: loaded from {os.path.basename(stats_path)}")
    with open(stats_path) as f:
        raw = json.load(f)

    # Extract normalization sub-dict and strip joint prefix
    norm = raw.get("normalization", raw)
    canonical = _canonicalize_stat_keys(norm)

    # Log training metadata if present
    if "best_val_rmse_nm" in raw:
        log_message(f"  Training RMSE: {raw['best_val_rmse_nm']:.4f} Nm")

    _log_norm_stats(canonical, os.path.basename(stats_path))
    return canonical


def _canonicalize_stat_keys(norm: dict) -> dict:
    """Strip joint prefix from stat keys to get canonical names.

    ``{"elbow_position": {...}, "elbow_velocity": {...}}``
    becomes ``{"position": {...}, "velocity": {...}}``.
    """
    canonical_suffixes = ["position_error", "position", "velocity", "torque"]
    result = {}
    for key, val in norm.items():
        for suffix in canonical_suffixes:
            if key.endswith(suffix):
                result[suffix] = val
                break
    return result


def _log_norm_stats(stats: dict, source: str):
    """Print normalization stats to CLI in a compact table."""
    keys = ["position", "position_error", "velocity", "torque"]
    header = f"  {'':>16s}  {'mean':>8s}  {'std':>8s}  {'p1':>8s}  {'p99':>8s}"
    log_message(header)
    for k in keys:
        s = stats.get(k)
        if s is None:
            continue
        log_message(
            f"  {k:>16s}  {s['mean']:>8.4f}  {s['std']:>8.4f}  "
            f"{s['p1']:>8.4f}  {s['p99']:>8.4f}"
        )


def _patch_lstm_compute(actuator, sysid_stats=None):
    """Updates ActuatorNetLSTM.compute() to fix Newton's joint indexing bug.

    Newton's ActuatorNetLSTM.compute() reads from shared full-robot data arrays
    (all joints) but sea_input is sized for this actuator's joints only. This
    patch properly indexes to the actuator's joints using self._joint_indices.

    Also fixes the output side: scatters computed torques to correct positions
    in the full-robot _computed_effort array (Newton's code replaces the whole
    array with a smaller one), and uses self._num_joints for the final Warp
    kernel dim so the boolean joint_mask indexes correctly.

    Supports two model types:
    - 3-input sysid model (position, position_error, velocity): uses sysid_stats
      (or global _SYSID_STATS fallback) for normalization with percentile clipping
    - 2-input model (pos_error, velocity): raw inputs, no normalization

    Args:
        actuator: Newton ActuatorNetLSTM instance to patch.
        sysid_stats: Optional normalization stats dict. Falls back to _SYSID_STATS.
    """
    # Detect RNN type and architecture from weight shapes.
    # Both LSTM and GRU wrappers register the RNN as 'lstm' (so Newton's init works),
    # so we distinguish by gate factor: LSTM has 4*hidden, GRU has 3*hidden.
    net_params = dict(actuator.network.named_parameters())

    input_size = net_params["lstm.weight_ih_l0"].shape[1]
    hidden_size = net_params["lstm.weight_hh_l0"].shape[1]
    gate_factor = net_params["lstm.weight_ih_l0"].shape[0] // hidden_size
    num_layers = sum(1 for k in net_params if k.startswith("lstm.weight_ih_l"))
    n = actuator._num_envs * actuator.num_joints

    rnn_type = "GRU" if gate_factor == 3 else "LSTM"
    log_message(
        f"{rnn_type} network: input_size={input_size}, hidden_size={hidden_size}, "
        f"num_layers={num_layers}, actuator_joints={actuator.num_joints}"
    )

    # Resize sea_input if the network expects a different input size than Newton allocated
    if actuator.sea_input.shape[2] != input_size:
        log_message(f"Resizing sea_input from {actuator.sea_input.shape[2]} to {input_size} channels")
        actuator.sea_input = torch.zeros(n, 1, input_size, device=actuator._device)

    # Resize hidden/cell state if num_layers differs
    if actuator.sea_hidden_state.shape[0] != num_layers:
        log_message(f"Resizing hidden state from {actuator.sea_hidden_state.shape[0]} to {num_layers} layers")
        actuator.sea_hidden_state = torch.zeros(num_layers, n, hidden_size, device=actuator._device)
        actuator.sea_cell_state = torch.zeros(num_layers, n, hidden_size, device=actuator._device)

    # --- Normalization setup ---
    use_sysid_norm = (input_size == 3)
    stats = sysid_stats or _SYSID_STATS

    if use_sysid_norm:
        # Sysid model: clip to [p1, p99] then z-normalize (matches training preprocessing)
        pos_mean = stats["position"]["mean"]
        pos_std = stats["position"]["std"]
        pos_p1 = stats["position"]["p1"]
        pos_p99 = stats["position"]["p99"]
        pos_err_mean = stats["position_error"]["mean"]
        pos_err_std = stats["position_error"]["std"]
        pos_err_p1 = stats["position_error"]["p1"]
        pos_err_p99 = stats["position_error"]["p99"]
        vel_mean = stats["velocity"]["mean"]
        vel_std = stats["velocity"]["std"]
        vel_p1 = stats["velocity"]["p1"]
        vel_p99 = stats["velocity"]["p99"]
        torque_mean = stats["torque"]["mean"]
        torque_std = stats["torque"]["std"]
        torque_p1 = stats["torque"]["p1"]
        torque_p99 = stats["torque"]["p99"]
        log_message("Patching with sysid normalization + clipping (3-input: position, position_error, velocity)")
    else:
        # 2-input model: [pos_error, velocity] without normalization
        log_message(f"Patching ActuatorNetLSTM to fix Newton joint indexing ({input_size}-input, no normalization)")

    def patched_compute(self):
        from isaaclab.utils.warp.update_kernels import update_array2D_with_array2D_masked

        # FIX: Index to actuator joints from full-robot data arrays
        indices = self._joint_indices
        pos_target = wp.to_torch(self.data._actuator_position_target)[:, indices]
        joint_pos = wp.to_torch(self.data._sim_bind_joint_pos)[:, indices]
        joint_vel = wp.to_torch(self.data._sim_bind_joint_vel)[:, indices]

        pos_error = (pos_target - joint_pos).flatten()
        vel = joint_vel.flatten()

        if use_sysid_norm:
            # 3-input sysid model: [position, position_error, velocity]
            # Clip to [p1, p99] then normalize — matches training preprocessing
            pos_clipped = joint_pos.flatten().clamp(pos_p1, pos_p99)
            pos_err_clipped = pos_error.clamp(pos_err_p1, pos_err_p99)
            vel_clipped = vel.clamp(vel_p1, vel_p99)
            self.sea_input[:, 0, 0] = (pos_clipped - pos_mean) / pos_std
            self.sea_input[:, 0, 1] = (pos_err_clipped - pos_err_mean) / pos_err_std
            self.sea_input[:, 0, 2] = (vel_clipped - vel_mean) / vel_std
        else:
            # 2-input model: [pos_error, velocity] (no normalization)
            self.sea_input[:, 0, 0] = pos_error
            self.sea_input[:, 0, 1] = vel

        # Run LSTM inference
        with torch.inference_mode():
            torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.network(
                self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
            )

        if use_sysid_norm:
            torques = torques * torque_std + torque_mean

        # FIX: Scatter to correct positions in full-robot array (wp.to_torch is zero-copy)
        computed_effort = wp.to_torch(self.data._computed_effort)
        computed_effort[:, indices] = torques.reshape(self._num_envs, self.num_joints)

        # Clip based on motor limits (DCMotor._clip_effort uses Warp kernel with mask)
        self._clip_effort(self.data._computed_effort, self.data._applied_effort)

        # FIX: Use self._num_joints (all joints) so boolean mask indexes correctly
        wp.launch(
            update_array2D_with_array2D_masked,
            dim=(self._num_envs, self._num_joints),
            inputs=[
                self.data._applied_effort,
                self.data.joint_effort,
                self._env_mask,
                self._joint_mask,
            ],
            device=self._device,
        )

    import types
    actuator.compute = types.MethodType(patched_compute, actuator)


# ---------------------------------------------------------------------------
# Arm actuator group helpers
# ---------------------------------------------------------------------------
_ARM_JOINT_EXPRS = [".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"]
_ARM_MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../input/actuator_models/h1")
_ACTUATOR_MODELS_BASE = os.path.join(os.path.dirname(__file__), "../../input/actuator_models")


def _arm_actuators_single_model(model_filename: str = "gru_fullarm_elbow_stateful_best.pt") -> dict:
    """One LSTM/GRU for all arm joints (default behavior)."""
    return {
        "arms": ActuatorNetLSTMCfg(
            joint_names_expr=_ARM_JOINT_EXPRS,
            network_file=_ensure_torchscript(os.path.join(_ARM_MODEL_DIR, model_filename)),
            saturation_effort=25.0,
            effort_limit=25.0,
            velocity_limit=13.5,
        ),
    }


def _arm_actuators_per_joint(model_files: dict[str, str] | None = None) -> dict:
    """Separate LSTM/GRU per joint type (left/right share same model).

    Args:
        model_files: Mapping from joint type to model filename, e.g.
            {"shoulder_pitch": "shoulder_pitch.pt", "elbow": "elbow.pt"}.
            Defaults to ``<joint_type>.pt`` for each joint type.
    """
    joint_types = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"]
    defaults = {jt: f"{jt}.pt" for jt in joint_types}
    files = model_files or defaults

    actuators = {}
    for jt in joint_types:
        fname = files.get(jt, f"{jt}.pt")
        actuators[f"arms_{jt}"] = ActuatorNetLSTMCfg(
            joint_names_expr=[f".*_{jt}"],
            network_file=_ensure_torchscript(os.path.join(_ARM_MODEL_DIR, fname)),
            saturation_effort=25.0,
            effort_limit=25.0,
            velocity_limit=13.5,
        )
    return actuators


# Per-robot arm joint config: joint regex patterns, actuator group name, model subdir
_ROBOT_ARM_CFG = {
    "h1": {
        "joint_exprs": [".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
        "group_name": "arms",
        "model_subdir": "h1",
    },
    "ur10e": {
        "joint_exprs": [".*"],
        "group_name": "arm",
        "model_subdir": "ur10e",
    },
}


def _build_arm_actuators(run_cfg: dict, robot_name: str) -> dict:
    """Build arm actuator config(s) from the run config's ``actuator`` section.

    Returns a dict like ``{"arms": <Cfg>}`` (single model) or
    ``{"arms_shoulder_pitch": ..., "arms_elbow": ...}`` (per-joint LSTM).
    Returns ``None`` if the run config has no ``actuator`` section (caller
    should keep the scene-class defaults).
    """
    act_cfg = run_cfg.get("actuator")
    if not act_cfg:
        return None

    model_type = act_cfg.get("model_type", "implicit")
    yaml_file = act_cfg.get("yaml_file")

    # Normalize: gru / gru_perjoint are aliases for lstm / lstm_perjoint
    # (both use ActuatorNetLSTMCfg — Isaac Lab handles LSTM and GRU the same way)
    _type_aliases = {"gru": "lstm", "gru_perjoint": "lstm_perjoint"}
    model_type = _type_aliases.get(model_type, model_type)

    arm = _ROBOT_ARM_CFG.get(robot_name)
    if arm is None:
        log_message(f"WARNING: no arm config for robot '{robot_name}', skipping actuator override")
        return None

    joint_exprs = arm["joint_exprs"]
    group_name = arm["group_name"]
    model_dir = os.path.join(_ACTUATOR_MODELS_BASE, arm["model_subdir"])

    if model_type == "implicit":
        if not yaml_file:
            raise ValueError("actuator.yaml_file required for model_type=implicit")
        return {group_name: load_implicit_actuator_cfg(yaml_file, joint_exprs)}

    elif model_type == "dcmotor":
        if not yaml_file:
            raise ValueError("actuator.yaml_file required for model_type=dcmotor")
        return {group_name: load_dc_motor_cfg(yaml_file, joint_exprs)}

    elif model_type == "lstm":
        network_file = act_cfg.get("network_file")
        if not network_file:
            raise ValueError("actuator.network_file required for model_type=lstm/gru")
        return {group_name: ActuatorNetLSTMCfg(
            joint_names_expr=joint_exprs,
            network_file=_ensure_torchscript(os.path.join(_ACTUATOR_MODELS_BASE, network_file)),
            saturation_effort=25.0,
            effort_limit=25.0,
            velocity_limit=13.5,
        )}

    elif model_type == "lstm_perjoint":
        network_files = act_cfg.get("network_files", {})
        if not network_files:
            raise ValueError("actuator.network_files dict required for model_type=lstm_perjoint/gru_perjoint")
        actuators = {}
        for jt, fname in network_files.items():
            actuators[f"{group_name}_{jt}"] = ActuatorNetLSTMCfg(
                joint_names_expr=[f".*_{jt}"],
                network_file=_ensure_torchscript(os.path.join(_ACTUATOR_MODELS_BASE, fname)),
                saturation_effort=25.0,
                effort_limit=25.0,
                velocity_limit=13.5,
            )
        return actuators

    elif model_type == "fmu":
        fmu_dir = act_cfg.get("fmu_path")
        if not fmu_dir:
            raise ValueError("actuator.fmu_path required for model_type=fmu")
        if not yaml_file:
            raise ValueError("actuator.yaml_file required for model_type=fmu (PD gains for torque_pred)")
        fmu_step = act_cfg.get("fmu_step_size", 0.002)
        return {group_name: load_fmu_actuator_cfg(
            yaml_file=yaml_file,
            fmu_path=fmu_dir,
            joint_names_expr=joint_exprs,
            fmu_step_size=fmu_step,
        )}

    else:
        raise ValueError(
            f"Unknown actuator model_type '{model_type}'. "
            "Choose from: implicit, dcmotor, lstm, lstm_perjoint, fmu"
        )


# ---------------------------------------------------------------------------
# Scene configuration for single-env benchmark
# ---------------------------------------------------------------------------

@configclass
class H1BenchmarkSceneCfg(InteractiveSceneCfg):
    """Scene with H1 robot and ground plane for motion benchmarking."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=H1_MINIMAL_CFG.spawn.replace(usd_path=_H1_LOCAL_USD, func=spawn_from_usd_with_fixed_base),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.5),
            joint_pos={
                ".*_hip_yaw": 0.0,
                ".*_hip_roll": 0.0,
                ".*_hip_pitch": -0.28,
                ".*_knee": 0.79,
                ".*_ankle": -0.52,
                "torso": 0.0,
                ".*_shoulder_pitch": 0.0,
                ".*_shoulder_roll": 0.0,
                ".*_shoulder_yaw": 0.0,
                ".*_elbow": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
                effort_limit_sim=300,
                stiffness={
                    ".*_hip_yaw": 50.0, ".*_hip_roll": 50.0,
                    ".*_hip_pitch": 100.0, ".*_knee": 100.0, "torso": 100.0,
                },
                damping={
                    ".*_hip_yaw": 5.0, ".*_hip_roll": 5.0,
                    ".*_hip_pitch": 5.0, ".*_knee": 5.0, "torso": 5.0,
                },
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle"],
                effort_limit_sim=100,
                stiffness={".*_ankle": 20.0},
                damping={".*_ankle": 4.0},
            ),
            # -----------------------------------------------------------------------------
            # Implicit actuator with real H1 motor params (from YAML)
            "arms": load_implicit_actuator_cfg(
                "h1/h1_arm_implicit.yaml",
                [".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            ),
            # -----------------------------------------------------------------------------
            # DC motor with velocity-dependent torque saturation (from YAML):
            # "arms": load_dc_motor_cfg(
            #     "h1_arm_dcmotor.yaml",
            #     [".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            # ),
            # -----------------------------------------------------------------------------
            # Explicit PD with built-in command delay (DelayedPDActuator).
            # Delays position/velocity commands by N physics steps before PD torque
            # computation. Uses motor_lag_ms from YAML. Note: this is *explicit* PD
            # (torques in Python), not implicit (physics engine).
            # "arms": load_delayed_pd_actuator_cfg(
            #     "h1_arm_implicit.yaml",
            #     [".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            # ),
            # -----------------------------------------------------------------------------
            # Distilled LSTM/GRU actuator net (learned from real H1 motor data).
            # Newton bug fix + optional normalization applied via _patch_lstm_compute().
            # Single model for all arm joints:
            # **_arm_actuators_single_model(),
            #
            # Per-joint models (one LSTM/GRU per joint type, L/R share same model):
            # **_arm_actuators_per_joint(),
            #
            # Per-joint with custom model files:
            # **_arm_actuators_per_joint({"shoulder_pitch": "sp.pt", "elbow": "elbow.pt", ...}),
        },
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class Ur10eBenchmarkSceneCfg(InteractiveSceneCfg):
    """Scene with UR10e robot for motion benchmarking."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=_UR10_LOCAL_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.712,
                "elbow_joint": 1.712,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
            },
        ),
        actuators={
            "arm": load_implicit_actuator_cfg(
                "ur10e/ur10e_implicit.yaml",
                [".*"],
            ),
        },
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# ---------------------------------------------------------------------------
# Robot-specific benchmark configurations
# ---------------------------------------------------------------------------
_BENCHMARK_ROBOT_CONFIGS = {
    "h1": {
        "scene_cfg_cls": H1BenchmarkSceneCfg,
        "actuator_yaml": "h1/h1_arm_implicit.yaml",
    },
    "ur10e": {
        "scene_cfg_cls": Ur10eBenchmarkSceneCfg,
        "actuator_yaml": "ur10e/ur10e_implicit.yaml",
    },
}


class NewtonJointMotionBenchmark:
    """Plays back joint motion trajectories under Newton physics and logs
    SAGE-compatible CSVs (control.csv, state_motor.csv, joint_list.txt).

    Uses InteractiveScene + SimulationContext with the same stepping pattern
    as ManagerBasedRLEnv (scene.write_data_to_sim / sim.step / scene.update).
    """

    def __init__(self, args):
        self.robot_name = args.robot_name.lower()
        self.motion_source = args.motion_source.lower()
        self.valid_joints_file = args.valid_joints_file
        self.output_folder = args.output_folder
        self.fix_root = args.fix_root
        self.physics_freq = args.physics_freq
        self.render_freq = args.render_freq
        self.original_control_freq = args.original_control_freq
        self.control_freq = args.control_freq or self.physics_freq
        self.kp = args.kp
        self.kd = args.kd
        self.record_video = args.record_video
        self.headless = args.headless

        # Load per-robot run config for solver/buffer settings
        from run_configs import load_run_cfg
        self._run_cfg = load_run_cfg(self.robot_name)

        # Resolve actuator YAML: run config > _BENCHMARK_ROBOT_CONFIGS fallback
        act_section = self._run_cfg.get("actuator", {})
        robot_cfg = _BENCHMARK_ROBOT_CONFIGS.get(self.robot_name, {})
        self._actuator_yaml = act_section.get("yaml_file") or robot_cfg.get("actuator_yaml", "")

        # Motor lag: CLI override > actuator YAML > 0
        cli_lag = getattr(args, "motor_lag_ms", None)
        if cli_lag is not None:
            self.motor_lag_ms = cli_lag
        elif self._actuator_yaml:
            actuator_params = load_actuator_params(self._actuator_yaml)
            self.motor_lag_ms = float(actuator_params.get("motor_lag_ms", 0.0))
        else:
            self.motor_lag_ms = 0.0

        # Frequency validation
        if self.physics_freq % self.render_freq != 0:
            raise ValueError("Physics frequency must be divisible by render frequency.")
        if self.render_freq % self.control_freq != 0:
            raise ValueError("Render frequency must be divisible by control frequency.")

        self.physics_dt = 1.0 / self.physics_freq
        self.render_dt = 1.0 / self.render_freq
        self.control_dt = 1.0 / self.control_freq
        self.divisor = self.render_freq // self.control_freq

        self.motor_lag_steps = round(self.motor_lag_ms / 1000.0 / self.physics_dt)

        self.set_valid_joints = False
        self.valid_joint_names = []
        self._load_valid_joints()

        self._setup_simulation()

    def _load_valid_joints(self):
        """Load valid joint names from config file."""
        if self.valid_joints_file is not None:
            config_file = self.valid_joints_file
        else:
            config_file = os.path.join(
                os.path.dirname(__file__), "configs", f"{self.robot_name}_valid_joints.txt"
            )

        if not os.path.isfile(config_file):
            self.set_valid_joints = False
            log_message(f"No valid joints file found: {config_file}, use all joints in motion file")
            return

        with open(config_file) as file:
            self.set_valid_joints = True
            self.valid_joint_names = [line.strip() for line in file if line.strip()]
        log_message(f"Loaded {len(self.valid_joint_names)} valid joint names from {config_file}")

    def _setup_simulation(self):
        """Initialize SimulationContext + InteractiveScene with Newton solver."""
        # Configure simulation with Newton solver
        sim_cfg = SimulationCfg(
            dt=self.physics_dt,
            render_interval=self.divisor,
            gravity=(0.0, 0.0, -9.81),
        )
        _sim_section = self._run_cfg.get("simulation", {})
        from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
        sim_cfg.physics = NewtonCfg(
            use_cuda_graph=False,
            solver_cfg=MJWarpSolverCfg(
                integrator=_sim_section.get("integrator", "implicitfast"),
                ls_iterations=_sim_section.get("ls_iterations", 10),
                iterations=_sim_section.get("solver_iterations", 10),
            ),
        )

        self.sim = SimulationContext(sim_cfg)

        # Build scene config from robot-specific class
        robot_cfg = _BENCHMARK_ROBOT_CONFIGS.get(self.robot_name)
        if robot_cfg is None:
            raise ValueError(
                f"Unknown robot '{self.robot_name}'. "
                f"Available: {list(_BENCHMARK_ROBOT_CONFIGS.keys())}"
            )
        scene_cfg = robot_cfg["scene_cfg_cls"](num_envs=1, env_spacing=4.0)

        # Override arm actuators from run config (actuator section)
        new_arm = _build_arm_actuators(self._run_cfg, self.robot_name)
        if new_arm is not None:
            # Remove old arm actuator key(s) from scene config
            old_keys = [k for k in scene_cfg.robot.actuators
                        if k in ("arms", "arm") or k.startswith("arms_")]
            for k in old_keys:
                del scene_cfg.robot.actuators[k]
            scene_cfg.robot.actuators.update(new_arm)

        # Apply CLI kp/kd overrides if provided (skip LSTM/GRU groups — they don't use kp/kd)
        if self.kp is not None:
            for group_name, cfg in scene_cfg.robot.actuators.items():
                if not isinstance(cfg, ActuatorNetLSTMCfg):
                    cfg.stiffness = self.kp
        if self.kd is not None:
            for group_name, cfg in scene_cfg.robot.actuators.items():
                if not isinstance(cfg, ActuatorNetLSTMCfg):
                    cfg.damping = self.kd

        # If no fixed root requested, revert spawn to default (no fixed joint).
        # Only applies to H1 — UR10e base is inherently fixed (bolted to table).
        if not self.fix_root and self.robot_name == "h1":
            scene_cfg.robot = scene_cfg.robot.replace(spawn=H1_MINIMAL_CFG.spawn.replace(usd_path=_H1_LOCAL_USD))

        # Create the scene (spawns all entities)
        self.scene = InteractiveScene(scene_cfg)

        # Reset simulation (triggers _initialize_impl which creates actuator instances)
        self.sim.reset()

        # Get robot articulation handle from scene
        self.robot = self.scene["robot"]

        # Patch LSTM/GRU actuators AFTER sim.reset() (actuators exist) but BEFORE first sim step.
        # Newton's ActuatorNetLSTM.compute() has a bug with joint indexing in multi-actuator setups.
        # Note: check by class name, not isinstance — Newton creates its own ActuatorNetLSTM
        # (isaaclab_newton.actuators.actuator_net.ActuatorNetLSTM) which is a different class
        # from the standard isaaclab.actuators.actuator_net.ActuatorNetLSTM we imported.
        for name, actuator in self.robot.actuators.items():
            if type(actuator).__name__ == "ActuatorNetLSTM":
                log_message(f"Patching '{name}' — model: {os.path.basename(actuator.cfg.network_file)}")
                stats = _load_sysid_stats(actuator.cfg.network_file)
                _patch_lstm_compute(actuator, sysid_stats=stats)

        # Reset scene state
        self.scene.reset()

        # Apply sysid friction parameters that Isaac Lab doesn't natively write
        # to the Newton solver (dynamic_friction → joint_friction,
        # viscous_friction → mujoco.dof_passive_damping).
        self._apply_newton_friction_params()

        # Build joint name-to-index mapping
        self._joint_name_to_idx = {}
        for i, name in enumerate(self.robot.joint_names):
            self._joint_name_to_idx[name] = i

        log_message(f"Newton simulation initialized with {len(self.robot.joint_names)} joints")
        log_message(f"Physics dt: {self.physics_dt}, Render dt: {self.render_dt}, Control dt: {self.control_dt}")
        if self.motor_lag_steps > 0:
            log_message(
                f"Motor command lag: {self.motor_lag_ms:.1f} ms = {self.motor_lag_steps} physics steps "
                f"({self.motor_lag_steps * self.physics_dt * 1000:.1f} ms effective)"
            )

        # Diagnostic: log actual actuator types and determine arms actuator suffix
        self._actuator_suffix = "unknown"
        arm_groups = []
        suffix_map = {
            "ImplicitActuator": "implicit",
            "DCMotor": "dcmotor",
            "ActuatorNetFMU": "fmu",
            "ActuatorNetLSTM": "lstm",
        }
        for name, actuator in self.robot.actuators.items():
            actuator_type = type(actuator).__name__
            log_message(f"Actuator '{name}': {actuator_type} (module: {type(actuator).__module__})")
            if name in ("arms", "arm") or name.startswith("arms_"):
                arm_groups.append(name)
                self._actuator_suffix = suffix_map.get(actuator_type, actuator_type.lower())
        if len(arm_groups) > 1:
            self._actuator_suffix += "_perjoint"

        # Use YAML filename stem as suffix when available (e.g. "h1_arm_sysid_implicit")
        act_section = self._run_cfg.get("actuator", {})
        yaml_file = act_section.get("yaml_file")
        if yaml_file:
            self._actuator_suffix = os.path.splitext(os.path.basename(yaml_file))[0]
        log_message(f"Arms actuator model: {self._actuator_suffix} (output folders will use this name)")

        self._log_motor_params()

        # Track simulation time manually
        self._sim_time = 0.0

    def _apply_newton_friction_params(self):
        """Write dynamic_friction and viscous_friction to Newton solver.

        Isaac Lab's actuator setup writes ``friction`` (static/Coulomb) to
        ``model.joint_friction`` but does NOT write:
        - ``dynamic_friction`` → should go to ``model.joint_friction``
        - ``viscous_friction`` → should go to ``model.mujoco.dof_passive_damping``

        This matches the workaround used in ``run_sysid.py``.
        """
        import re

        # Collect per-joint friction values from arm actuator configs
        has_dynamic = False
        has_viscous = False
        for _name, actuator in self.robot.actuators.items():
            cfg = actuator.cfg
            if getattr(cfg, "dynamic_friction", None) is not None:
                has_dynamic = True
            if getattr(cfg, "viscous_friction", None) is not None:
                has_viscous = True

        if not has_dynamic and not has_viscous:
            return

        # Locate the Newton model via BFS from robot.root_view
        newton_model = None
        visited = set()
        queue = [("view", self.robot.root_view)]
        for _depth in range(4):
            next_queue = []
            for path, obj in queue:
                oid = id(obj)
                if oid in visited:
                    continue
                visited.add(oid)
                if hasattr(obj, "joint_target_ke"):
                    newton_model = obj
                    log_message(f"Newton model found at root_view -> {path}")
                    break
                for attr in dir(obj):
                    if attr.startswith("_"):
                        continue
                    try:
                        child = getattr(obj, attr)
                        if hasattr(child, "__dict__") or hasattr(child, "__slots__"):
                            next_queue.append((f"{path}.{attr}", child))
                    except Exception:
                        pass
            if newton_model is not None:
                break
            queue = next_queue

        if newton_model is None:
            log_message("WARNING: Could not find Newton model — skipping friction params")
            return

        device = self.robot.device
        num_dofs = len(self.robot.joint_names)

        def _resolve_param(param_value, joint_names):
            """Resolve a scalar or regex-keyed dict to per-joint values."""
            if isinstance(param_value, (int, float)):
                return [float(param_value)] * len(joint_names)
            if isinstance(param_value, dict):
                vals = [0.0] * len(joint_names)
                for i, jname in enumerate(joint_names):
                    for pattern, v in param_value.items():
                        if re.fullmatch(pattern, jname):
                            vals[i] = float(v)
                            break
                return vals
            return [0.0] * len(joint_names)

        # Build per-DOF arrays for dynamic_friction and viscous_friction
        dynamic_vals = np.zeros(num_dofs)
        viscous_vals = np.zeros(num_dofs)

        for _name, actuator in self.robot.actuators.items():
            cfg = actuator.cfg
            joint_ids = actuator.joint_indices
            if isinstance(joint_ids, slice):
                joint_ids = list(range(*joint_ids.indices(num_dofs)))

            joint_names = [self.robot.joint_names[j] for j in joint_ids]

            if has_dynamic and getattr(cfg, "dynamic_friction", None) is not None:
                vals = _resolve_param(cfg.dynamic_friction, joint_names)
                for j, v in zip(joint_ids, vals):
                    dynamic_vals[j] = v

            if has_viscous and getattr(cfg, "viscous_friction", None) is not None:
                vals = _resolve_param(cfg.viscous_friction, joint_names)
                for j, v in zip(joint_ids, vals):
                    viscous_vals[j] = v

        # Write dynamic_friction → model.joint_friction (Coulomb friction)
        if has_dynamic and np.any(dynamic_vals > 0):
            joint_friction = newton_model.joint_friction
            friction_np = joint_friction.numpy()
            # Newton model may be (num_worlds, num_dofs) or (total_dofs,)
            if friction_np.ndim == 1:
                friction_np[:num_dofs] = dynamic_vals
            else:
                for w in range(friction_np.shape[0]):
                    friction_np[w, :num_dofs] = dynamic_vals
            joint_friction.assign(friction_np)
            log_message(f"Applied dynamic_friction to Newton joint_friction: {dynamic_vals[dynamic_vals > 0].tolist()}")

        # Write viscous_friction → model.mujoco.dof_passive_damping
        if has_viscous and np.any(viscous_vals > 0):
            mujoco_ns = getattr(newton_model, "mujoco", None)
            if mujoco_ns is None or not hasattr(mujoco_ns, "dof_passive_damping"):
                log_message("WARNING: Newton model has no mujoco.dof_passive_damping — viscous_friction not applied")
                return
            dof_damping = mujoco_ns.dof_passive_damping
            dof_np = dof_damping.numpy()
            if dof_np.ndim == 1:
                dof_np[:num_dofs] = viscous_vals
            else:
                for w in range(dof_np.shape[0]):
                    dof_np[w, :num_dofs] = viscous_vals
            dof_damping.assign(dof_np)
            log_message(f"Applied viscous_friction to Newton dof_passive_damping: {viscous_vals[viscous_vals > 0].tolist()}")

    def _log_motor_params(self):
        """Print actuator parameters for each group in a table."""
        log_message("\nActuator Parameters:")
        log_message("=" * 90)
        for name, actuator in self.robot.actuators.items():
            cfg = actuator.cfg
            actuator_type = type(actuator).__name__
            log_message(f"  [{name}] {actuator_type}")
            log_message(f"  {'Parameter':<30} {'Value':<20} {'Unit'}")
            log_message(f"  {'-' * 84}")

            # Common params from cfg
            params = [
                ("stiffness (kp)", cfg.stiffness, "N-m/rad"),
                ("damping (kd)", cfg.damping, "N-m-s/rad"),
                ("effort_limit", cfg.effort_limit, "N-m"),
                ("effort_limit_sim", cfg.effort_limit_sim, "N-m"),
                ("velocity_limit", cfg.velocity_limit, "rad/s"),
                ("velocity_limit_sim", cfg.velocity_limit_sim, "rad/s"),
                ("armature", cfg.armature, "kg-m^2"),
                ("friction", cfg.friction, ""),
            ]

            # Newton-specific params (set post-construction)
            for attr in ("dynamic_friction", "viscous_friction"):
                val = getattr(cfg, attr, None)
                if val is not None:
                    params.append((attr, val, ""))

            # DCMotor-specific
            if hasattr(cfg, "saturation_effort"):
                params.append(("saturation_effort", cfg.saturation_effort, "N-m"))

            # LSTM-specific
            if hasattr(cfg, "network_file"):
                params.append(("network_file", os.path.basename(cfg.network_file), ""))

            for pname, val, unit in params:
                if val is not None:
                    log_message(f"  {pname:<30} {str(val):<20} {unit}")

            log_message("")

        # Motor lag (benchmark-level, not actuator-level)
        log_message(f"  {'motor_lag_ms':<30} {self.motor_lag_ms:<20.1f} {'ms'}")
        if self.motor_lag_steps > 0:
            log_message(
                f"  {'motor_lag_steps':<30} {self.motor_lag_steps:<20} "
                f"physics steps ({self.motor_lag_steps * self.physics_dt * 1000:.1f} ms)"
            )
        log_message("=" * 90)

    def _to_torch(self, data):
        """Convert wp.array to torch tensor (Newton uses Warp arrays, not PyTorch)."""
        if isinstance(data, wp.array):
            return wp.to_torch(data)
        return data

    def _sim_step(self):
        """Perform one physics step using the ManagerBasedRLEnv pattern."""
        self.scene.write_data_to_sim()
        self.sim.step(render=not self.headless)
        self._sim_time += self.physics_dt
        self.scene.update(self.physics_dt)

    def set_motion(self, motion_file, motion_name):
        """Set up for a new motion file."""
        self.motion_file = motion_file
        self.motion_name = motion_name

        # Parse joint names from motion file header
        with open(self.motion_file) as file:
            first_line = file.readline().strip()

        all_joint_names = [name.strip() for name in first_line.split(",")]

        # Filter to valid joints
        if self.set_valid_joints:
            self.joint_names = [j for j in all_joint_names if j in self.valid_joint_names]
        else:
            self.joint_names = all_joint_names

        if not self.joint_names:
            raise ValueError("No valid joints found in motion file that match the config file")

        log_message(f"Filtered {len(self.joint_names)} valid joints from motion file")

        # Map joint names to Isaac Lab articulation indices
        self.joint_indices = []
        for joint in self.joint_names:
            key = joint.split("/")[-1]
            if key in self._joint_name_to_idx:
                self.joint_indices.append(self._joint_name_to_idx[key])
            else:
                log_message(f"WARNING: Joint '{key}' not found in robot, skipping")
        self.joint_indices = torch.tensor(self.joint_indices, dtype=torch.long, device=self.robot.device)

        self._init_logger()

    def _load_motion(self):
        """Load motion data from file, optionally resampling."""
        with open(self.motion_file) as file:
            lines = file.readlines()

        all_joint_names = [name.strip() for name in lines[0].strip().split(",")]

        # Find indices of valid joints in the motion file
        valid_indices = []
        for joint in self.joint_names:
            idx = all_joint_names.index(joint)
            valid_indices.append(idx)

        # Parse joint values
        joint_angles = [[] for _ in range(len(valid_indices))]
        for line in lines[1:]:
            if not line.strip():
                continue
            values = [val.strip() for val in line.strip().split(",")]
            if len(values) == len(all_joint_names):
                for i, valid_idx in enumerate(valid_indices):
                    if values[valid_idx]:
                        joint_angles[i].append(float(values[valid_idx]))

        # Resample if needed
        # joint_angles layout: joint_angles[joint_idx] = [frame0, frame1, ...]
        # i.e. shape (num_joints, num_frames)
        if self.original_control_freq is not None and self.original_control_freq != self.control_freq:
            log_message(f"Resampling motion from {self.original_control_freq}Hz to {self.control_freq}Hz")
            joint_angles = np.array(joint_angles)  # shape: (num_joints, num_frames)
            n_joints, old_len = joint_angles.shape

            duration = old_len / self.original_control_freq
            new_len = int(round(duration * self.control_freq))

            old_times = np.linspace(0, duration, old_len, endpoint=False)
            new_times = np.linspace(0, duration, new_len, endpoint=False)
            new_times = new_times[new_times <= old_times[-1]]
            new_times = np.append(new_times, old_times[-1])
            new_len = len(new_times)

            new_angles = np.zeros((n_joints, new_len), dtype=joint_angles.dtype)
            for i in range(n_joints):
                f = interp1d(old_times, joint_angles[i, :], kind="linear")
                new_angles[i, :] = f(new_times)
            joint_angles = new_angles.tolist()

        return joint_angles, self.joint_names

    def _init_logger(self):
        """Create output directory and CSV files in SAGE-compatible format."""
        self.sim_output_folder = os.path.join(
            self.output_folder, "sim", self.robot_name, self.motion_source,
            self._actuator_suffix, self.motion_name,
        )
        os.makedirs(self.sim_output_folder, exist_ok=True)

        self.joint_list_file = os.path.join(self.sim_output_folder, "joint_list.txt")
        self.control_file = os.path.join(self.sim_output_folder, "control.csv")
        self.dof_file = os.path.join(self.sim_output_folder, "state_motor.csv")

        # Write joint list
        with open(self.joint_list_file, "w") as file:
            for joint in self.joint_names:
                file.write(f"{joint}\n")

        # Initialize control log
        with open(self.control_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "timestamp", "positions"])

        # Initialize state motor log
        with open(self.dof_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])

    def _log_state(self, time, command_positions, actual_positions, actual_velocities, actual_efforts):
        """Log current robot state to CSV in SAGE-compatible format.

        Timestamps are converted to microseconds to match the real robot's
        SAGE format (convert_h1_chirp_to_csv.py writes timestamps in µs).
        """
        ts_us = time * 1e6  # seconds -> microseconds
        with open(self.control_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["CONTROL", f"{ts_us:.1f}", command_positions.tolist()])

        with open(self.dof_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["STATE_MOTOR", f"{ts_us:.1f}", actual_positions.tolist(), actual_velocities.tolist(), actual_efforts.tolist()]
            )

    def run_benchmark(self):
        """Run motion playback benchmark under Newton physics."""
        joint_angles, _ = self._load_motion()
        num_steps = len(joint_angles[0])
        num_joints = len(self.joint_names)

        log_message(f"Loading motion data from {self.motion_file}...")
        log_message(f"Physics dt: {self.physics_dt}, Control dt: {self.control_dt}, Divisor: {self.divisor}")

        # Note: sim.reset() is only called once in _setup_simulation().
        # Newton's solver_cfg is consumed on first reset and cannot be re-initialized.
        self._sim_time = 0.0

        # --- Buffer phase: interpolate from current pose to motion start ---
        _bench_section = self._run_cfg.get("benchmark", {})
        BUFFER_TIME = float(_bench_section.get("buffer_time", 5.0))
        buffer_control_steps = int(BUFFER_TIME / self.control_dt)

        joint_pos = self._to_torch(self.robot.data.joint_pos)
        initial_pos = joint_pos[0, self.joint_indices].cpu().numpy()
        motion_start_pos = np.array([joint_angles[j][0] for j in range(num_joints)])

        log_message(f"Starting initialization phase with {BUFFER_TIME}s buffer...")

        for step in range(buffer_control_steps * self.divisor):
            ctrl_step = step // self.divisor

            if step % self.divisor == 0:
                alpha = ctrl_step / buffer_control_steps
                interp_pos = (1 - alpha) * initial_pos + alpha * motion_start_pos

                joint_pos = self._to_torch(self.robot.data.joint_pos)
                target = joint_pos.clone()
                target[0, self.joint_indices] = torch.tensor(
                    interp_pos, dtype=torch.float32, device=self.robot.device
                )
                self.robot.set_joint_position_target(target)

            self._sim_step()

        buffer_end_time = self._sim_time

        joint_pos = self._to_torch(self.robot.data.joint_pos)
        current_pos = joint_pos[0, self.joint_indices].cpu().numpy()
        log_message(
            f"Buffer completed in {BUFFER_TIME:.2f}s. "
            f"Joint positions set to (rad): {list(current_pos)}"
        )
        log_message("Initialization complete. Starting main motion...")

        # --- Command delay buffer ---
        # Delays position targets by self.motor_lag_steps physics steps to model
        # communication/actuation latency. Pre-fill with the motion start pose so
        # the first motor_lag_steps steps hold the robot at the starting position.
        lag = self.motor_lag_steps
        if lag > 0:
            cmd_buffer = deque(maxlen=lag + 1)
            for _ in range(lag):
                cmd_buffer.append(motion_start_pos.copy())
        else:
            cmd_buffer = None

        # --- Main motion playback ---
        for counter in range(num_steps * self.divisor):
            index = counter // self.divisor
            if index >= num_steps:
                break

            adjusted_time = self._sim_time - buffer_end_time

            # Set joint targets at control frequency
            if counter % self.divisor == 0:
                current_cmd = np.array([joint_angles[j_idx][index] for j_idx in range(num_joints)])

                if cmd_buffer is not None:
                    # Push current command, pop delayed command
                    cmd_buffer.append(current_cmd)
                    delayed_cmd = cmd_buffer[0]
                else:
                    delayed_cmd = current_cmd

                joint_pos = self._to_torch(self.robot.data.joint_pos)
                target = joint_pos.clone()
                for j_idx, art_idx in enumerate(self.joint_indices):
                    target[0, art_idx] = delayed_cmd[j_idx]
                self.robot.set_joint_position_target(target)

            # Read state and log (wp.array -> torch -> numpy for CSV)
            # Log the original (non-delayed) command for comparison with real data
            cmd_pos = np.array([joint_angles[k][index] for k in range(num_joints)])
            act_pos = self._to_torch(self.robot.data.joint_pos)[0, self.joint_indices].cpu().numpy()
            act_vel = self._to_torch(self.robot.data.joint_vel)[0, self.joint_indices].cpu().numpy()

            # Read applied torques (motor torque from PD + friction).
            # Note: real H1 torque sensors report motor torque, not joint
            # torque, so gravity compensation should NOT be added here.
            try:
                act_eff = self._to_torch(self.robot.data.applied_torque)[0, self.joint_indices].cpu().numpy()
            except (AttributeError, RuntimeError):
                act_eff = np.zeros(num_joints)

            self._log_state(adjusted_time, cmd_pos, act_pos, act_vel, act_eff)

            self._sim_step()

        final_pos = self._to_torch(self.robot.data.joint_pos)[0, self.joint_indices].cpu().numpy()
        log_message(
            f"Motion completed in {counter + 1} physics steps. "
            f"Joint positions stopped at (rad): {list(final_pos)}"
        )
        log_message(f"Benchmark complete for {self.motion_name}. Results saved to {self.sim_output_folder}")

    def log_joint_properties(self):
        """Print a summary of joint properties."""
        joint_names = self.robot.joint_names
        log_message("\nJoint Properties Summary:")
        log_message("=" * 80)
        log_message(f"{'Joint Name':<40} {'Index':<10}")
        log_message("-" * 80)
        for i, name in enumerate(joint_names):
            log_message(f"{name:<40} {i:<10}")
        log_message("=" * 80)

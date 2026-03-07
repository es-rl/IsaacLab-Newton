"""CMA-ES optimizer for actuator system identification.

Adapted from PACE (ETH Zurich) — operates in normalized [-1, 1] space with
linear mapping to physical parameter bounds.  Uses the lightweight `cmaes`
Python package (same as PACE).
"""

import csv
import os

import cmaes as cmaes_lib
import torch
import yaml


class CMAESOptimizer:
    """CMA-ES optimizer that maps a flat parameter vector to per-joint physics properties.

    Parameter layout (for N_types joint types and P properties):
        [0 : N_types]          first property  (e.g. armature)
        [N_types : 2*N_types]  second property (e.g. dynamic_friction)
        ...
    Total dimension = N_types * P.

    Mirroring (default, H1): the 4 joint-type values are expanded to 8 per-joint
    values (right joints first, then left joints with the same values).
    No mirroring (UR10e): N joint-type values map directly to N joints.
    Controlled by ``mirror`` flag in the bounds YAML (default: true).
    """

    def __init__(
        self,
        config_path: str,
        num_envs: int,
        device: str,
        sigma: float | None = None,
        max_iterations: int | None = None,
        epsilon: float | None = None,
        joint_types: list[str] | None = None,
    ):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.joint_types: list[str] = joint_types if joint_types is not None else cfg["joint_types"]
        self.num_types = len(self.joint_types)
        self.mirror: bool = cfg.get("mirror", True)
        self.num_envs = num_envs
        self.device = device

        # Build parameter names, bounds, and slices
        self.param_names: list[str] = []
        bounds_lower: list[float] = []
        bounds_upper: list[float] = []
        self.property_names: list[str] = []

        for prop_name, prop_cfg in cfg["parameters"].items():
            self.property_names.append(prop_name)
            for jt in self.joint_types:
                self.param_names.append(f"{prop_name}/{jt}")
                bounds_lower.append(float(prop_cfg["lower"]))
                bounds_upper.append(float(prop_cfg["upper"]))

        self.num_params = len(self.param_names)
        self.bounds = torch.tensor(
            list(zip(bounds_lower, bounds_upper)), dtype=torch.float32, device=device
        )  # (num_params, 2)

        # Fixed params
        self.fixed = cfg.get("fixed", {})

        # CMA-ES settings (CLI overrides > YAML)
        cmaes_cfg = cfg.get("cmaes", {})
        self.sigma = sigma if sigma is not None else float(cmaes_cfg.get("sigma", 0.5))
        self.max_iterations = (
            max_iterations if max_iterations is not None else int(cmaes_cfg.get("max_iterations", 200))
        )
        self.epsilon = epsilon if epsilon is not None else float(cmaes_cfg.get("epsilon", 0.01))

        # Initialize CMA-ES in normalized [-1, 1] space
        import numpy as np

        mean_norm = np.zeros(self.num_params)
        bounds_norm = np.array([[-1.0, 1.0]] * self.num_params)
        self.cma = cmaes_lib.CMA(
            mean=mean_norm,
            sigma=self.sigma,
            bounds=bounds_norm,
            seed=0,
            population_size=num_envs,
        )

        # Candidate parameters for current generation (physical space)
        self.params = torch.zeros(num_envs, self.num_params, device=device)
        # Normalized params (for CMA-ES tell)
        self._params_norm = torch.zeros(num_envs, self.num_params, device=device)

        # Score accumulation
        self.scores = torch.zeros(num_envs, device=device)
        self._score_steps = 0

        self.generation = 0
        self._best_score = float("inf")
        self._best_params = None

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _denormalize(self, norm_params: torch.Tensor) -> torch.Tensor:
        """Map from [-1, 1] to physical bounds."""
        t = (norm_params + 1.0) / 2.0  # [0, 1]
        return self.bounds[:, 0] + t * (self.bounds[:, 1] - self.bounds[:, 0])

    # ------------------------------------------------------------------
    # Population sampling
    # ------------------------------------------------------------------

    def sample_population(self):
        """Ask CMA-ES for a new generation of candidates.

        Returns:
            Tensor of shape (num_envs, num_params) in physical parameter space.
        """
        for i in range(self.num_envs):
            x = torch.tensor(self.cma.ask(), dtype=torch.float32, device=self.device)
            self._params_norm[i] = x
            self.params[i] = self._denormalize(x)

        # Reset scores for this generation
        self.scores.zero_()
        self._score_steps = 0
        return self.params

    # ------------------------------------------------------------------
    # Score accumulation
    # ------------------------------------------------------------------

    def accumulate_score(self, sim_pos: torch.Tensor, real_pos: torch.Tensor):
        """Accumulate per-step squared position error.

        Args:
            sim_pos: (num_envs, num_arm_joints) simulated joint positions.
            real_pos: (num_arm_joints,) or (1, num_arm_joints) real joint positions.
        """
        if real_pos.dim() == 1:
            real_pos = real_pos.unsqueeze(0)
        diff = sim_pos - real_pos
        self.scores += (diff * diff).sum(dim=1)  # sum over joints
        self._score_steps += 1

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve(self) -> bool:
        """Report scores to CMA-ES and advance one generation.

        Returns:
            True if converged (should stop).
        """
        if self._score_steps > 0:
            mean_scores = self.scores / self._score_steps
        else:
            mean_scores = self.scores

        # Tell CMA-ES
        solutions = []
        for i in range(self.num_envs):
            solutions.append((self._params_norm[i].cpu().numpy().tolist(), mean_scores[i].item()))
        self.cma.tell(solutions)

        # Track best
        best_idx = mean_scores.argmin().item()
        if mean_scores[best_idx] < self._best_score:
            self._best_score = mean_scores[best_idx].item()
            self._best_params = self.params[best_idx].clone()

        self.generation += 1

        # Convergence check
        score_range = mean_scores.max() - mean_scores.min()
        if mean_scores.min() > 0:
            rel_range = score_range / mean_scores.min()
        else:
            rel_range = score_range

        converged = self.generation >= self.max_iterations or (
            self.epsilon is not None and rel_range < self.epsilon
        )
        return converged

    # ------------------------------------------------------------------
    # Parameter expansion (4 joint types → 8 per-joint)
    # ------------------------------------------------------------------

    def expand_to_joints(self, candidate: torch.Tensor, prop_name: str) -> torch.Tensor:
        """Extract one property from a candidate vector and expand to per-joint values.

        With mirroring (H1): returns (2*N_types,) — right then left (same values).
        Without mirroring (UR10e): returns (N_types,) — values as-is.
        """
        prop_idx = self.property_names.index(prop_name)
        start = prop_idx * self.num_types
        values = candidate[start : start + self.num_types]
        if self.mirror:
            return values.repeat(2)
        return values

    def expand_all_envs(self, prop_name: str) -> torch.Tensor:
        """Expand one property for all envs.

        With mirroring: returns (num_envs, 2*N_types).
        Without mirroring: returns (num_envs, N_types).
        """
        prop_idx = self.property_names.index(prop_name)
        start = prop_idx * self.num_types
        values = self.params[:, start : start + self.num_types]
        if self.mirror:
            return values.repeat(1, 2)
        return values

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_best_params(self, joint_name_map: dict[str, str] | None = None) -> dict:
        """Return best parameters as a dict suitable for YAML output.

        Args:
            joint_name_map: Optional mapping from sysid joint type names to
                actual sim joint names (e.g. {"shoulder_pan": "shoulder_pan_joint"}).
                If provided, output keys use the mapped names so the result can
                be pasted directly into actuator YAML files.
        """
        if self._best_params is None:
            return {}

        result = {}
        for prop_name in self.property_names:
            prop_idx = self.property_names.index(prop_name)
            start = prop_idx * self.num_types
            values = self._best_params[start : start + self.num_types]
            result[prop_name] = {}
            for i, jt in enumerate(self.joint_types):
                key = joint_name_map.get(jt, jt) if joint_name_map else jt
                result[prop_name][key] = round(float(values[i]), 6)
            # Also store a single average for easy copy to YAML
            result[f"{prop_name}_mean"] = round(float(values.mean()), 6)
        result["best_mse"] = self._best_score
        result.update(self.fixed)
        return result

    def log_generation(self, log_file: str):
        """Append one row to the optimization CSV log."""
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["generation", "best_mse", "mean_mse", "min_mse"] + self.param_names
                writer.writerow(header)

            if self._score_steps > 0:
                mean_scores = self.scores / self._score_steps
            else:
                mean_scores = self.scores

            best_idx = mean_scores.argmin().item()
            row = [
                self.generation,
                self._best_score,
                mean_scores.mean().item(),
                mean_scores[best_idx].item(),
            ]
            row += [round(float(v), 6) for v in self.params[best_idx]]
            writer.writerow(row)

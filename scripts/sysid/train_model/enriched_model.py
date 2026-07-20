# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-step force residual GRU for differentiable MuJoCo training."""

from __future__ import annotations

import torch
import torch.nn as nn


class ForceResidualGRU(nn.Module):
    """Predicts per-joint residual forces from joint state.

    Input:  (B, T, 8) — [position_error(4), velocity(4)]
    Output: (B, T, 4) — residual forces bounded ±force_bound Nm
    """

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_joints: int = 4,
        force_bound: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_joints = num_joints
        self.force_bound = force_bound

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.force_head = nn.Linear(hidden_size, num_joints)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, T, 8) input features.
            h: (num_layers, B, hidden_size) hidden state, or None to zero-init.

        Returns:
            forces: (B, T, 4) bounded residual forces.
            h_new:  (num_layers, B, hidden_size) updated hidden state.
        """
        out, h_new = self.gru(x, h)
        forces = self.force_head(out)
        forces = torch.tanh(forces) * self.force_bound
        return forces, h_new

    def init_hidden(self, batch_size: int = 1, device: str = "cuda") -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

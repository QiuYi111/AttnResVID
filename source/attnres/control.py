"""
Control variants for Attention-Residual mechanism.

This module implements control variants for ablation studies:
- C1 ConcatFusionBlock: Concatenation-based fusion (no attention)
- C2 GateOnlyBlock: Only learned gate, no history aggregation
"""

import torch
import torch.nn as nn
from typing import List


class ConcatFusionBlock(nn.Module):
    """Concatenation-based fusion block (C1 Control).

    This control variant concatenates historical feature maps along
    the channel dimension and uses a 1x1 convolution to project
    back to the original channel count. This tests whether the
    attention mechanism provides benefits over simple concatenation.

    Args:
        num_channels: Number of channels in feature maps
        history_len: Maximum number of historical feature maps
    """

    def __init__(self, num_channels: int, history_len: int):
        super().__init__()
        self.num_channels = num_channels
        self.history_len = history_len

        # Project concatenated features back to original channels
        # Create a separate projection for each possible history length (1 to history_len)
        # to handle variable available history at runtime
        self.projs = nn.ModuleList([
            nn.Conv2d(
                num_channels * (i + 1),  # residual + i history features
                num_channels,
                kernel_size=1,
            )
            for i in range(1, history_len + 1)
        ])

    def forward(
        self,
        current: torch.Tensor,
        history: List[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenation-based fusion.

        Args:
            current: Current feature map (B, C, H, W)
            history: List of historical feature maps
            residual: Standard residual from conv layer (B, C, H, W)

        Returns:
            Fused output (B, C, H, W)
        """
        # Take last history_len features
        history = history[-self.history_len:] if len(history) > self.history_len else history

        if not history:
            return residual

        # Concatenate along channel dimension
        all_features = [residual] + list(history)
        concat_feat = torch.cat(all_features, dim=1)  # (B, C*(N+1), H, W)

        # Use appropriate projection based on actual history length
        proj_idx = len(history) - 1  # 0 for 1 history, 1 for 2 history, etc.
        return self.projs[proj_idx](concat_feat)


class GateOnlyBlock(nn.Module):
    """Gate-only block (C2 Control).

    This control variant only applies a learnable gate to the standard
    residual connection, without any history aggregation. This tests
    whether the benefits come from the gating mechanism or the
    attention-based history aggregation.

    Args:
        num_channels: Number of channels
        gate_type: "scalar" or "channel" gating
        init_value: Initial gate value
    """

    def __init__(
        self,
        num_channels: int,
        gate_type: str = "scalar",
        init_value: float = 0.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.gate_type = gate_type

        if gate_type == "scalar":
            self.gate = nn.Parameter(torch.full((), init_value))
        else:
            self.gate = nn.Parameter(torch.full((num_channels,), init_value))

    def forward(
        self,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gate to residual.

        Args:
            residual: Standard residual (B, C, H, W)

        Returns:
            Gated residual (B, C, H, W)
        """
        if self.gate_type == "scalar":
            weight = torch.sigmoid(self.gate)
            return residual * weight
        else:
            weight = torch.sigmoid(self.gate).view(1, -1, 1, 1)
            return residual * weight

    def get_gate_value(self) -> float:
        """Get current gate value."""
        with torch.no_grad():
            if self.gate_type == "scalar":
                return torch.sigmoid(self.gate).item()
            else:
                return torch.sigmoid(self.gate).mean().item()


class BaselineResidual(nn.Module):
    """Baseline residual block (no AttnRes).

    This simply returns the standard residual without any modification.
    Used to ensure identical behavior when AttnRes is disabled.
    """

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """Return residual unchanged."""
        return residual

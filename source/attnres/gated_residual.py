"""
Gated Attention-Residual mechanism.

This module implements the core GatedAttnResidual class that combines
standard residual connections with attention-aggregated historical features.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Literal

from .config import AttnResConfig
from .aggregator import DepthAttentionAggregator, LearnedGate


class GatedAttnResidual(nn.Module):
    """Gated Attention-Residual block.

    This module enhances standard residual connections by:
    1. Aggregating historical feature maps using depth-wise attention
    2. Combining the result with the current residual via a learnable gate

    The forward pass computes:
        residual = standard_conv(x)
        attn_features = aggregate(history)
        gated_attn = gate(attn_features)
        output = activation(x + residual + gated_attn)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        config: AttnResConfig containing all hyperparameters
        block_idx: Index of this block in the model
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: AttnResConfig,
        block_idx: int = 0,
        shared_aggregator: Optional["DepthAttentionAggregator"] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = config
        self.block_idx = block_idx

        # Determine effective history length for this block
        self.history_len = config.get_effective_history_len(block_idx)

        # Create attention aggregator (or use shared one)
        if config.fusion_mode == "attention":
            if shared_aggregator is not None:
                # Share aggregator across blocks (no own parameters for aggregation)
                self.aggregator = shared_aggregator
                self._owns_aggregator = False
            else:
                self.aggregator = DepthAttentionAggregator(
                    num_channels=out_channels,
                    history_len=self.history_len,
                    temperature=config.temperature,
                    score_fn=config.score_fn,
                    detach_history=config.detach_history,
                )
                self._owns_aggregator = True
        else:
            self.aggregator = None
            self._owns_aggregator = False

        # Create learnable gate
        if config.fusion_mode in ["attention", "gate_only"]:
            self.gate = LearnedGate(
                num_channels=out_channels,
                gate_type=config.gate_type,
                init_value=config.gate_init,
            )
        else:
            self.gate = None

        # For concat mode, we need a projection layer
        if config.fusion_mode == "concat":
            # In concat mode, we need to handle variable history length.
            # We create projection layers for each possible history length (0 to history_len).
            # History length 0 means only residual (no concat needed).
            self.concat_projs = nn.ModuleList([
                nn.Conv2d(
                    out_channels * (i + 1),  # residual + i history features
                    out_channels,
                    kernel_size=1,
                )
                for i in range(1, self.history_len + 1)  # 1 to history_len
            ])

    def forward(
        self,
        x: torch.Tensor,
        history: List[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gated attention-residual.

        Args:
            x: Input tensor (B, C, H, W)
            history: List of historical feature maps, each (B, C, H, W)
            residual: Standard residual from conv layer (B, C, H, W)

        Returns:
            Output tensor with gated AttnRes applied (B, C, H, W)
        """
        if self.config.fusion_mode == "attention":
            # Aggregate history using attention
            # Note: Current implementation aggregates only history features.
            # To include current layer x_l in attention (as per README formula),
            # uncomment the following line and update aggregator call:
            # attn_feat = self.aggregator(history, current=x)
            history = history[-self.history_len:] if len(history) > self.history_len else history

            if history:
                attn_feat = self.aggregator(history, current=x)
                gated_attn = self.gate(attn_feat)
                return residual + gated_attn
            else:
                # No history yet, use current directly (self-attention)
                attn_feat = x
                gated_attn = self.gate(attn_feat)
                return residual + gated_attn

        elif self.config.fusion_mode == "gate_only":
            # Only apply gate, no history aggregation (C2 control)
            # This tests whether the benefits come from the gating mechanism
            # rather than the history aggregation.
            # Preserves residual path: output = residual + gated_residual
            gated = self.gate(residual)
            return residual + gated

        elif self.config.fusion_mode == "concat":
            # Concatenate history and project (C1 control)
            history = history[-self.history_len:] if len(history) > self.history_len else history

            if history:
                # Concatenate along channel dimension
                # residual: (B, C, H, W), history: [(B, C, H, W), ...]
                all_features = [residual] + list(history)
                concat_feat = torch.cat(all_features, dim=1)  # (B, C*(N+1), H, W)
                # Use the appropriate projection based on actual history length
                proj_idx = len(history) - 1  # 0 for 1 history, 1 for 2 history, etc.
                return self.concat_projs[proj_idx](concat_feat)
            else:
                return residual

        else:
            return residual

    def get_info(self) -> dict:
        """Get information about this block for logging."""
        info = {
            "block_idx": self.block_idx,
            "fusion_mode": self.config.fusion_mode,
            "history_len": self.history_len,
        }

        if self.gate is not None:
            info["gate_value"] = self.gate.get_gate_value()

        return info

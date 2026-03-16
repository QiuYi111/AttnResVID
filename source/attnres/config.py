"""
Configuration for Attention-Residual (AttnRes) mechanism.

The AttnResConfig dataclass contains all parameters needed to configure
the attention-based residual learning for DeepVIDv2.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class AttnResConfig:
    """Configuration for Attention-Residual mechanism.

    Attributes:
        enabled: Whether AttnRes is enabled (False = baseline behavior)
        mode: Which blocks use AttnRes
            - "off": Disabled (same as enabled=False)
            - "bottleneck": Only last 2 bottleneck blocks
            - "bottleneck_decoder": Last 2 blocks + decoder
            - "stagewise": All blocks with stage-wise history
        history_len: Number of previous feature maps to aggregate
        temperature: Softmax temperature for attention (lower = sharper)
        gate_init: Initial value for learnable gate (0 = start as identity)
        score_fn: Method to compute attention scores
            - "gap_linear": Global avg pooling -> linear projection
            - "conv1x1_gap_linear": Conv1x1 -> GAP -> linear
        gate_type: Type of gating
            - "scalar": Single learned scalar per block
            - "channel": Per-channel learned gates
        detach_history: Detach history from computation graph (stability)
        fusion_mode: How to combine features
            - "attention": Weighted sum using attention weights
            - "concat": Concatenation followed by conv (C1 control)
            - "gate_only": Only gate, no history (C2 control)
        share_proj: Whether to share projection parameters across blocks
        bottleneck_start_idx: Index where bottleneck blocks start (default 2/4)
        decoder_enabled: Whether to extend to decoder blocks
    """

    enabled: bool = False
    mode: Literal["off", "bottleneck", "bottleneck_decoder", "stagewise"] = "off"
    history_len: int = 2
    temperature: float = 1.0
    gate_init: float = 0.0
    score_fn: Literal["gap_linear", "conv1x1_gap_linear"] = "gap_linear"
    gate_type: Literal["scalar", "channel"] = "scalar"
    detach_history: bool = False
    fusion_mode: Literal["attention", "concat", "gate_only"] = "attention"
    share_proj: bool = False
    bottleneck_start_idx: int = 2
    decoder_enabled: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.mode == "off":
            self.enabled = False
        elif self.enabled and self.mode == "off":
            self.mode = "bottleneck"  # Default to bottleneck if enabled

        if self.history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {self.history_len}")

        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

        if self.gate_init < 0 or self.gate_init > 1:
            raise ValueError(f"gate_init must be in [0, 1], got {self.gate_init}")

    @classmethod
    def from_args(cls, args) -> "AttnResConfig":
        """Create config from argparse arguments.

        This allows easy integration with DeepVIDv2's CLI.
        """
        enabled = getattr(args, "attnres_enabled", False)
        if not enabled:
            return cls(enabled=False)

        return cls(
            enabled=True,
            mode=getattr(args, "attnres_mode", "bottleneck"),
            history_len=getattr(args, "attnres_history_len", 2),
            temperature=getattr(args, "attnres_temperature", 1.0),
            gate_init=getattr(args, "attnres_gate_init", 0.0),
            score_fn=getattr(args, "attnres_score_fn", "gap_linear"),
            gate_type=getattr(args, "attnres_gate_type", "scalar"),
            detach_history=getattr(args, "attnres_detach_history", False),
            fusion_mode=getattr(args, "attnres_fusion_mode", "attention"),
            share_proj=getattr(args, "attnres_share_proj", False),
            bottleneck_start_idx=getattr(args, "attnres_bottleneck_start_idx", 2),
            decoder_enabled=getattr(args, "attnres_decoder_enabled", False),
        )

    def should_use_attnres(self, block_idx: int, total_blocks: int) -> bool:
        """Determine if a block should use AttnRes based on mode and index.

        Args:
            block_idx: Index of the block in the model
            total_blocks: Total number of blocks in the model

        Returns:
            True if this block should use AttnRes
        """
        if not self.enabled:
            return False

        if self.mode == "bottleneck":
            return block_idx >= self.bottleneck_start_idx
        elif self.mode == "bottleneck_decoder":
            return block_idx >= self.bottleneck_start_idx
        elif self.mode == "stagewise":
            return True
        return False

    def get_effective_history_len(self, block_idx: int) -> int:
        """Get effective history length for a specific block.

        Early blocks may have less history available.

        Args:
            block_idx: Index of the block

        Returns:
            Effective history length for this block
        """
        if self.mode == "stagewise":
            return min(self.history_len, block_idx)
        return self.history_len

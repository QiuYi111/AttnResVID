"""
Wrapper for integrating AttnRes with existing ResBlocks.

This module provides the AttnResBlockWrapper class that wraps standard
ResBlock instances to add Attention-Residual functionality.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .config import AttnResConfig
from .gated_residual import GatedAttnResidual
from .history_manager import StageHistoryManager


class AttnResDecoderWrapper(nn.Module):
    """Wrapper for decoder blocks (ConvBlock) to add Attention-Residual functionality.

    This wraps a standard ConvBlock (decoder block) and adds AttnRes functionality.

    Args:
        block: The original ConvBlock to wrap
        config: AttnResConfig containing all hyperparameters
        block_idx: Index of this decoder block
        num_channels: Number of channels for AttnRes modules
        use_attnres: Whether this block should use AttnRes
    """

    def __init__(
        self,
        block: nn.Module,
        config: AttnResConfig,
        block_idx: int,
        num_channels: int,
        use_attnres: bool = False,
    ):
        super().__init__()
        self.block = block
        self.config = config
        self.block_idx = block_idx
        self.num_channels = num_channels
        self.use_attnres = use_attnres

        # Create AttnRes components if enabled
        if self.use_attnres:
            self.attnres = GatedAttnResidual(
                in_channels=num_channels,
                out_channels=num_channels,
                config=config,
                block_idx=block_idx,
            )
        else:
            self.attnres = None

    def forward(
        self,
        x: torch.Tensor,
        history: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass with optional AttnRes.

        Args:
            x: Input tensor (B, C, H, W)
            history: List of historical feature maps (for AttnRes)

        Returns:
            Output tensor (B, C, H, W)
        """
        # Run original block
        out = self.block(x)

        # Apply AttnRes if enabled and history is available
        if self.use_attnres and self.attnres is not None and history:
            # For decoder blocks, the residual is the output itself
            # since decoder blocks don't have skip connections
            attn_feat = self.attnres.aggregator(history, current=out)
            gated_attn = self.attnres.gate(attn_feat)
            out = out + gated_attn

        return out

    def get_attnres_info(self) -> dict:
        """Get information about AttnRes for this block."""
        if self.attnres is not None:
            info = self.attnres.get_info()
            info["block_type"] = "decoder"
            return info
        return {"use_attnres": False, "block_type": "decoder"}


class AttnResBlockWrapper(nn.Module):
    """Wrapper for ResBlock to add Attention-Residual functionality.

    This wraps a standard ResBlock and optionally adds AttnRes functionality
    based on the configuration. The wrapper preserves the original block's
    behavior when AttnRes is disabled.

    Args:
        block: The original ResBlock to wrap
        config: AttnResConfig containing all hyperparameters
        block_idx: Index of this block in the model
        num_channels: Number of channels for AttnRes modules
        use_attnres: Whether this block should use AttnRes (determined by caller)
    """

    def __init__(
        self,
        block: nn.Module,
        config: AttnResConfig,
        block_idx: int,
        num_channels: int,
        use_attnres: bool = False,
    ):
        super().__init__()
        self.block = block
        self.config = config
        self.block_idx = block_idx
        self.num_channels = num_channels
        self.use_attnres = use_attnres

        # Create AttnRes components if enabled
        if self.use_attnres:
            self.attnres = GatedAttnResidual(
                in_channels=num_channels,
                out_channels=num_channels,
                config=config,
                block_idx=block_idx,
            )
        else:
            self.attnres = None

    def forward(
        self,
        x: torch.Tensor,
        history: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass with optional AttnRes.

        Args:
            x: Input tensor (B, C, H, W)
            history: List of historical feature maps (for AttnRes)

        Returns:
            Output tensor (B, C, H, W)
        """
        # Run original block
        out = self.block(x)

        # Apply AttnRes if enabled and history is available
        if self.use_attnres and self.attnres is not None and history:
            # Compute standard residual (out - x)
            # Note: out already includes the residual from the block
            # We need to compute what the residual contribution was
            residual = out - x
            attn_output = self.attnres(x, history, residual)
            out = x + attn_output

        return out

    def get_attnres_info(self) -> dict:
        """Get information about AttnRes for this block."""
        if self.attnres is not None:
            return self.attnres.get_info()
        return {"use_attnres": False}


class AttnResModelWrapper(nn.Module):
    """Wrapper for the entire DeepVIDv2 model with AttnRes.

    This class manages the history across all blocks and provides
    a clean interface for the forward pass.

    Args:
        model: The original DeepVIDv2 model
        config: AttnResConfig containing all hyperparameters
    """

    def __init__(
        self,
        model: nn.Module,
        config: AttnResConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.history_manager = StageHistoryManager(config)

        # Wrap each block if needed
        if config.enabled:
            self._wrap_blocks()

    def _wrap_blocks(self):
        """Wrap ResBlocks with AttnResBlockWrapper."""
        num_blocks = len(self.model.model)
        wrapped_blocks = nn.ModuleList()

        for i, block in enumerate(self.model.model):
            use_attnres = self.config.should_use_attnres(i, num_blocks)
            if use_attnres:
                wrapped = AttnResBlockWrapper(
                    block=block,
                    config=self.config,
                    block_idx=i,
                    num_channels=self.model.num_feature,
                    use_attnres=True,
                )
                wrapped_blocks.append(wrapped)
            else:
                wrapped_blocks.append(block)

        self.model.model = wrapped_blocks

    def forward(self, x: torch.Tensor, encode: bool = False):
        """Forward pass with history management.

        Args:
            x: Input tensor (B, C, H, W)
            encode: Whether to return feature list

        Returns:
            Output tensor (and optionally feature list)
        """
        # Clear history at the start of each forward pass
        self.history_manager.clear()

        out = self.model.in_block(x)
        identity = out
        feat_list = []

        num_blocks = len(self.model.model)

        for i, block in enumerate(self.model.model):
            # Get history for this block
            history = []
            if self.config.enabled:
                history = self.history_manager.get_history(i, num_blocks)

            # Forward pass
            if isinstance(block, AttnResBlockWrapper):
                out = block(out, history)
            else:
                out = block(out)

            # Add to history for future blocks
            if self.config.enabled:
                self.history_manager.add_feature(out, i, num_blocks)

            feat_list.append(out)

        out = self.model.pre_out_block(out)
        out += identity
        out = self.model.out_block(out)

        if encode:
            return out, feat_list
        else:
            return out

    def get_attnres_info(self) -> dict:
        """Get information about all AttnRes blocks."""
        info = {
            "config": {
                "enabled": self.config.enabled,
                "mode": self.config.mode,
                "history_len": self.config.history_len,
                "temperature": self.config.temperature,
                "fusion_mode": self.config.fusion_mode,
            },
            "blocks": [],
        }

        for i, block in enumerate(self.model.model):
            if isinstance(block, AttnResBlockWrapper):
                block_info = block.get_attnres_info()
                block_info["block_idx"] = i
                info["blocks"].append(block_info)

        return info

    def get_gate_values(self) -> dict:
        """Get current gate values for all AttnRes blocks."""
        gate_values = {}

        for i, block in enumerate(self.model.model):
            if isinstance(block, AttnResBlockWrapper):
                if block.attnres is not None and block.attnres.gate is not None:
                    gate_values[f"block_{i}"] = block.attnres.gate.get_gate_value()

        return gate_values

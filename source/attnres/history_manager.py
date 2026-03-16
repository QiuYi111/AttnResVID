"""
History management for Attention-Residual mechanism.

This module manages feature map history across blocks in the DeepVIDv2 network.
"""

import torch
from typing import List, Dict, Optional

from .config import AttnResConfig


class StageHistoryManager:
    """Manages feature history for blocks using AttnRes.

    This class tracks feature maps produced by each block and provides
    them to subsequent blocks for attention-based aggregation.

    The manager supports different modes:
    - bottleneck: Only last blocks share history
    - stagewise: Each stage maintains separate history
    - global: All blocks share global history

    Args:
        config: AttnResConfig containing history settings
    """

    def __init__(self, config: AttnResConfig):
        self.config = config
        self.histories: Dict[str, List[torch.Tensor]] = {}
        self._stage_names: List[str] = []

    def get_stage_name(self, block_idx: int, total_blocks: int) -> str:
        """Get the stage name for a given block.

        Args:
            block_idx: Index of the block
            total_blocks: Total number of blocks

        Returns:
            Stage name string
        """
        if self.config.mode == "stagewise":
            # In stagewise mode, all blocks share a single "stagewise" stage.
            # This allows history sharing across consecutive blocks as per exp.md §6.3.
            return "stagewise"
        elif self.config.mode in ["bottleneck", "bottleneck_decoder"]:
            if block_idx < self.config.bottleneck_start_idx:
                return "encoder"
            else:
                return "bottleneck"
        else:
            return "global"

    def initialize_stage(self, stage_name: str):
        """Initialize a stage's history if not exists.

        Args:
            stage_name: Name of the stage
        """
        if stage_name not in self.histories:
            self.histories[stage_name] = []
            if stage_name not in self._stage_names:
                self._stage_names.append(stage_name)

    def add_feature(
        self,
        feature: torch.Tensor,
        block_idx: int,
        total_blocks: int,
    ):
        """Add a feature map to the appropriate stage history.

        Args:
            feature: Feature map tensor (B, C, H, W)
            block_idx: Index of the producing block
            total_blocks: Total number of blocks
        """
        stage_name = self.get_stage_name(block_idx, total_blocks)
        self.initialize_stage(stage_name)

        # Add feature to history
        self.histories[stage_name].append(feature)

        # Trim to max history length
        max_len = self.config.history_len
        if len(self.histories[stage_name]) > max_len:
            self.histories[stage_name] = self.histories[stage_name][-max_len:]

    def get_history(
        self,
        block_idx: int,
        total_blocks: int,
    ) -> List[torch.Tensor]:
        """Get history for a specific block.

        Args:
            block_idx: Index of the requesting block
            total_blocks: Total number of blocks

        Returns:
            List of historical feature maps
        """
        stage_name = self.get_stage_name(block_idx, total_blocks)
        self.initialize_stage(stage_name)
        return self.histories[stage_name].copy()

    def get_effective_history_len(
        self,
        block_idx: int,
        total_blocks: int,
    ) -> int:
        """Get effective history length for a block.

        Early blocks may have less history available.

        Args:
            block_idx: Index of the block
            total_blocks: Total number of blocks

        Returns:
            Effective history length
        """
        history = self.get_history(block_idx, total_blocks)
        return len(history)

    def clear(self):
        """Clear all histories."""
        self.histories.clear()
        self._stage_names.clear()

    def get_stages(self) -> List[str]:
        """Get list of all stage names."""
        return self._stage_names.copy()

    def get_stage_history(self, stage_name: str) -> List[torch.Tensor]:
        """Get history for a specific stage.

        Args:
            stage_name: Name of the stage

        Returns:
            List of historical feature maps
        """
        return self.histories.get(stage_name, []).copy()

    def __len__(self) -> int:
        """Get total number of stages."""
        return len(self._stage_names)

    def __repr__(self) -> str:
        return f"StageHistoryManager(stages={self._stage_names}, mode={self.config.mode})"


class GlobalHistoryManager:
    """Simple global history manager for all blocks.

    This is a simpler alternative that maintains a single global history
    for all blocks. Use this for the "stagewise" mode.

    Args:
        max_len: Maximum history length
    """

    def __init__(self, max_len: int = 2):
        self.max_len = max_len
        self.history: List[torch.Tensor] = []

    def add(self, feature: torch.Tensor):
        """Add a feature map to history.

        Args:
            feature: Feature map tensor (B, C, H, W)
        """
        self.history.append(feature)
        if len(self.history) > self.max_len:
            self.history = self.history[-self.max_len:]

    def get(self) -> List[torch.Tensor]:
        """Get current history."""
        return self.history.copy()

    def clear(self):
        """Clear history."""
        self.history.clear()

    def __len__(self) -> int:
        return len(self.history)

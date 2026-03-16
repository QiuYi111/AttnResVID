"""
Aggregator modules for Attention-Residual mechanism.

This module implements depth-wise attention aggregation for CNN feature maps,
adapting Transformer-style attention to 2D spatial features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Literal


class RMSNorm2d(nn.Module):
    """Root Mean Square Normalization for 2D feature maps.

    Unlike BatchNorm, this normalizes across channels for each spatial location.
    Unlike LayerNorm, this operates on the channel dimension independently.

    Formula: output = input / sqrt(mean(input^2) + eps) * weight
    """

    def __init__(self, num_channels: int, eps: float = 1e-8):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Normalized tensor of same shape
        """
        # Compute variance over channel dimension
        variance = x.pow(2).mean(dim=1, keepdim=True)  # (B, 1, H, W)
        x_norm = x / torch.sqrt(variance + self.eps)
        return self.weight.view(1, -1, 1, 1) * x_norm


class DepthAttentionAggregator(nn.Module):
    """Channel-wise depth attention for 2D CNN feature maps.

    This module aggregates historical feature maps using learnable attention
    weights. It adapts Transformer-style sequence attention to work with
    2D CNN feature maps.

    The key idea is to:
    1. Compute a channel descriptor for each feature map via Global Average Pooling
    2. Normalize descriptors using RMS normalization
    3. Compute attention scores via dot product with a learned query
    4. Aggregate features using softmax-weighted sum

    Args:
        num_channels: Number of channels in feature maps
        history_len: Maximum number of historical feature maps to aggregate
        temperature: Softmax temperature (lower = sharper attention)
        score_fn: Method to compute attention scores
        detach_history: Whether to detach history from computation graph
    """

    def __init__(
        self,
        num_channels: int,
        history_len: int,
        temperature: float = 1.0,
        score_fn: Literal["gap_linear", "conv1x1_gap_linear"] = "gap_linear",
        detach_history: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.history_len = history_len
        self.temperature = temperature
        self.score_fn = score_fn
        self.detach_history = detach_history

        # Learned query vector for computing attention scores
        # Initialize with small random values for stability
        query = torch.empty(num_channels)
        nn.init.normal_(query, mean=0.0, std=0.02)
        self.query = nn.Parameter(query)

        # Normalization for channel descriptors
        self.norm = RMSNorm2d(num_channels)

        # Optional conv1x1 for alternative scoring
        if score_fn == "conv1x1_gap_linear":
            self.conv1x1 = nn.Conv2d(num_channels, num_channels, kernel_size=1)

    def compute_channel_descriptor(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute channel descriptor via Global Average Pooling.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Channel descriptor of shape (B, C)
        """
        if self.score_fn == "conv1x1_gap_linear":
            x = self.conv1x1(x)

        # Global Average Pooling
        desc = x.mean(dim=(2, 3))  # (B, C)
        return desc

    def compute_attention_scores(
        self, history: List[torch.Tensor], current: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention scores for historical feature maps.

        Args:
            history: List of historical feature maps, each (B, C, H, W)
            current: Current feature map (optional, for scoring)

        Returns:
            Attention scores of shape (B, len(history) + optional_current)
        """
        batch_size = history[0].shape[0]
        device = history[0].device

        descriptors = []

        # Process history
        for feat in history:
            if self.detach_history:
                feat = feat.detach()

            desc = self.compute_channel_descriptor(self.norm(feat))
            descriptors.append(desc)

        # Optionally include current
        if current is not None:
            desc = self.compute_channel_descriptor(self.norm(current))
            descriptors.append(desc)

        # Stack descriptors: (B, num_features, C)
        descriptors = torch.stack(descriptors, dim=1)  # (B, N, C)

        # Compute scores via dot product with learned query
        # query: (C,), descriptors: (B, N, C) -> scores: (B, N)
        scores = torch.einsum("c,bnc->bn", self.query, descriptors)

        # Normalize by temperature and apply softmax
        scores = scores / self.temperature
        weights = F.softmax(scores, dim=1)  # (B, N)

        return weights

    def forward(
        self,
        history: List[torch.Tensor],
        current: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Aggregate historical feature maps using attention.

        Args:
            history: List of historical feature maps, each (B, C, H, W)
            current: Current feature map (optional, included in aggregation)

        Returns:
            Aggregated feature map of shape (B, C, H, W)
        """
        if not history and current is None:
            raise ValueError("At least one of history or current must be provided")

        if current is None:
            # Aggregate only history
            weights = self.compute_attention_scores(history)
            # Stack history: (N, B, C, H, W) -> (B, N, C, H, W)
            stacked = torch.stack(history, dim=1)
            # weights: (B, N) -> (B, N, 1, 1, 1)
            weights = weights.view(-1, len(history), 1, 1, 1)
            # Weighted sum: (B, C, H, W)
            output = (stacked * weights).sum(dim=1)
        else:
            # Aggregate history + current
            weights = self.compute_attention_scores(history, current)
            # Stack history + current: (N+1, B, C, H, W) -> (B, N+1, C, H, W)
            all_features = history + [current]
            stacked = torch.stack(all_features, dim=1)
            # weights: (B, N+1) -> (B, N+1, 1, 1, 1)
            weights = weights.view(-1, len(all_features), 1, 1, 1)
            # Weighted sum: (B, C, H, W)
            output = (stacked * weights).sum(dim=1)

        return output

    def get_attention_weights(
        self,
        history: List[torch.Tensor],
        current: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return attention weights for analysis.

        Args:
            history: List of historical feature maps
            current: Current feature map (optional)

        Returns:
            Attention weights of shape (B, N)
        """
        with torch.no_grad():
            return self.compute_attention_scores(history, current)


class LearnedGate(nn.Module):
    """Learnable gate for controlling AttnRes contribution.

    The gate allows the model to smoothly interpolate between:
    - No AttnRes (gate = 0): Standard residual connection
    - Full AttnRes (gate = 1): Attention-aggregated features

    Args:
        num_channels: Number of channels (for channel-wise gating)
        gate_type: "scalar" for single gate, "channel" for per-channel
        init_value: Initial gate value (0 = start as identity)
    """

    def __init__(
        self,
        num_channels: int,
        gate_type: Literal["scalar", "channel"] = "scalar",
        init_value: float = 0.0,
    ):
        super().__init__()
        self.gate_type = gate_type

        if gate_type == "scalar":
            self.gate = nn.Parameter(torch.full((), init_value))
        else:  # channel
            self.gate = nn.Parameter(torch.full((num_channels,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gate to input.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Gated tensor of same shape
        """
        if self.gate_type == "scalar":
            # Scalar gate: use sigmoid for smooth interpolation
            weight = torch.sigmoid(self.gate)
            return x * weight
        else:
            # Channel-wise gate
            weight = torch.sigmoid(self.gate).view(1, -1, 1, 1)
            return x * weight

    def get_gate_value(self) -> float:
        """Get current gate value (for monitoring)."""
        with torch.no_grad():
            if self.gate_type == "scalar":
                return torch.sigmoid(self.gate).item()
            else:
                return torch.sigmoid(self.gate).mean().item()

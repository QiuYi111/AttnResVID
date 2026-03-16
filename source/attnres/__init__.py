"""
Attention-Residual (AttnRes) module for DeepVIDv2.

This module implements learnable depth-wise feature aggregation for
low-photon voltage imaging denoising.

Components:
- config: Configuration dataclass
- aggregator: Depth-wise attention for CNN feature maps
- gated_residual: Gated attention mechanism
- history_manager: Feature history tracking
- wrapper: Block wrapper for integration
- control: Control fusion variants
"""

from .config import AttnResConfig
from .aggregator import DepthAttentionAggregator, RMSNorm2d, LearnedGate
from .gated_residual import GatedAttnResidual
from .history_manager import StageHistoryManager, GlobalHistoryManager
from .wrapper import AttnResBlockWrapper, AttnResModelWrapper
from .control import ConcatFusionBlock, GateOnlyBlock, BaselineResidual

__all__ = [
    "AttnResConfig",
    "DepthAttentionAggregator",
    "RMSNorm2d",
    "LearnedGate",
    "GatedAttnResidual",
    "StageHistoryManager",
    "GlobalHistoryManager",
    "AttnResBlockWrapper",
    "AttnResModelWrapper",
    "ConcatFusionBlock",
    "GateOnlyBlock",
    "BaselineResidual",
]

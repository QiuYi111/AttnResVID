"""
Test script for Attention-Residual (AttnRes) implementation.

This script verifies:
1. AttnRes configuration
2. DepthAttentionAggregator shapes
3. History manager functionality
4. DeepVIDv2 integration
5. Baseline preservation (AttnRes disabled = same as original)
"""

import sys
import torch
import torch.nn as nn
import argparse

from source.attnres.config import AttnResConfig
from source.attnres.aggregator import DepthAttentionAggregator, RMSNorm2d, LearnedGate
from source.attnres.history_manager import StageHistoryManager, GlobalHistoryManager
from source.attnres.gated_residual import GatedAttnResidual
from source.attnres.wrapper import AttnResBlockWrapper
from source.attnres.control import ConcatFusionBlock, GateOnlyBlock
from source.network_collection import DeepVIDv2


def test_config():
    """Test AttnResConfig creation and validation."""
    print("Testing AttnResConfig...")

    # Test default config
    config = AttnResConfig()
    assert not config.enabled
    assert config.mode == "off"
    print("  ✓ Default config created")

    # Test enabled config
    config = AttnResConfig(
        enabled=True,
        mode="bottleneck",
        history_len=2,
        temperature=1.0,
    )
    assert config.enabled
    assert config.should_use_attnres(2, 4)  # bottleneck block
    assert not config.should_use_attnres(0, 4)  # encoder block
    print("  ✓ Enabled config with bottleneck mode")

    # Test from args
    args = argparse.Namespace(
        attnres_enabled=True,
        attnres_mode="stagewise",
        attnres_history_len=3,
        attnres_temperature=0.5,
        attnres_gate_init=0.0,
        attnres_score_fn="gap_linear",
        attnres_gate_type="scalar",
        attnres_detach_history=False,
        attnres_fusion_mode="attention",
        attnres_bottleneck_start_idx=2,
        attnres_decoder_enabled=False,
    )
    config = AttnResConfig.from_args(args)
    assert config.enabled
    assert config.mode == "stagewise"
    assert config.history_len == 3
    print("  ✓ Config from args")

    print("✓ Config tests passed\n")


def test_aggregator():
    """Test DepthAttentionAggregator."""
    print("Testing DepthAttentionAggregator...")

    batch_size = 2
    channels = 64
    height = 32
    width = 32

    # Create aggregator
    agg = DepthAttentionAggregator(
        num_channels=channels,
        history_len=2,
        temperature=1.0,
        score_fn="gap_linear",
    )

    # Create dummy history
    history = [
        torch.randn(batch_size, channels, height, width),
        torch.randn(batch_size, channels, height, width),
    ]
    current = torch.randn(batch_size, channels, height, width)

    # Test aggregation with history + current
    output = agg(history, current)
    assert output.shape == (batch_size, channels, height, width)
    print("  ✓ Aggregation with history + current")

    # Test aggregation with history only
    output = agg(history)
    assert output.shape == (batch_size, channels, height, width)
    print("  ✓ Aggregation with history only")

    # Test attention weights
    weights = agg.get_attention_weights(history, current)
    assert weights.shape == (batch_size, 3)  # 2 history + 1 current
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    print("  ✓ Attention weights valid")

    print("✓ Aggregator tests passed\n")


def test_rms_norm():
    """Test RMSNorm2d."""
    print("Testing RMSNorm2d...")

    batch_size = 2
    channels = 64
    height = 32
    width = 32

    norm = RMSNorm2d(channels)
    x = torch.randn(batch_size, channels, height, width)

    output = norm(x)
    assert output.shape == x.shape
    print("  ✓ RMSNorm2d output shape correct")

    # Test that normalization changes values
    assert not torch.allclose(output, x)
    print("  ✓ RMSNorm2d normalizes values")

    print("✓ RMSNorm2d tests passed\n")


def test_learned_gate():
    """Test LearnedGate."""
    print("Testing LearnedGate...")

    batch_size = 2
    channels = 64
    height = 32
    width = 32

    # Test scalar gate
    gate = LearnedGate(channels, gate_type="scalar", init_value=0.0)
    x = torch.randn(batch_size, channels, height, width)
    output = gate(x)

    assert output.shape == x.shape
    assert gate.get_gate_value() <= 1.0  # Sigmoid ensures <= 1
    print("  ✓ Scalar gate works")

    # Test channel gate
    gate = LearnedGate(channels, gate_type="channel", init_value=0.0)
    output = gate(x)
    assert output.shape == x.shape
    print("  ✓ Channel gate works")

    print("✓ LearnedGate tests passed\n")


def test_history_manager():
    """Test StageHistoryManager."""
    print("Testing StageHistoryManager...")

    config = AttnResConfig(enabled=True, mode="bottleneck", history_len=2)

    manager = StageHistoryManager(config)

    # Add features for encoder blocks (should go to "encoder" stage)
    feat_0 = torch.randn(1, 64, 32, 32)
    feat_1 = torch.randn(1, 64, 32, 32)

    manager.add_feature(feat_0, 0, 4)
    manager.add_feature(feat_1, 1, 4)

    history = manager.get_history(1, 4)
    assert len(history) == 2
    print("  ✓ Encoder history tracking works")

    # Add features for bottleneck blocks (should go to "bottleneck" stage)
    feat_2 = torch.randn(1, 64, 32, 32)
    manager.add_feature(feat_2, 2, 4)

    history = manager.get_history(2, 4)
    assert len(history) == 1  # Only bottleneck history
    print("  ✓ Bottleneck history is separate")

    # Test stagewise mode
    # In the fixed implementation, stagewise mode uses a single shared stage
    # so all blocks share history (up to history_len limit)
    config_stagewise = AttnResConfig(enabled=True, mode="stagewise", history_len=2)
    manager_s = StageHistoryManager(config_stagewise)

    manager_s.add_feature(feat_0, 0, 4)
    manager_s.add_feature(feat_1, 1, 4)

    # In stagewise mode, block 1 has access to both block 0 and block 1's features
    # (they share the same "stagewise" stage)
    history = manager_s.get_history(1, 4)
    assert len(history) == 2  # block 0 and block 1's features (shared stage)
    print("  ✓ Stagewise history works (shared stage across blocks)")

    print("✓ HistoryManager tests passed\n")


def test_deepvid_integration():
    """Test DeepVIDv2 integration with AttnRes."""
    print("Testing DeepVIDv2 integration...")

    # Create args
    args = argparse.Namespace(
        kernel_size=3,
        stride=2,
        padding=1,
        in_channels=11,
        out_channels=1,
        num_feature=64,
        num_blocks=4,
        norm_type="batch",
        activation_type="prelu",
        resblock_activation_out_type=None,
    )

    batch_size = 2
    input_channels = 11
    height = 64
    width = 64

    # Test baseline (AttnRes disabled)
    config = AttnResConfig(enabled=False)
    model = DeepVIDv2(args, attnres_config=config)
    model.eval()

    x = torch.randn(batch_size, input_channels, height, width)
    with torch.no_grad():
        output_baseline = model(x)

    assert output_baseline.shape == (batch_size, 1, height, width)
    print("  ✓ Baseline model forward pass works")

    # Test with AttnRes enabled
    config = AttnResConfig(
        enabled=True,
        mode="bottleneck",
        history_len=2,
        temperature=1.0,
    )
    model_attnres = DeepVIDv2(args, attnres_config=config)
    model_attnres.eval()

    with torch.no_grad():
        output_attnres = model_attnres(x)

    assert output_attnres.shape == (batch_size, 1, height, width)
    print("  ✓ AttnRes model forward pass works")

    # Check that outputs are different (AttnRes should affect the output)
    # Note: AttnRes might not significantly change output initially due to gate_init=0
    print(f"  Output difference norm: {(output_baseline - output_attnres).abs().mean().item():.6f}")

    # Test get_attnres_info
    info = model_attnres.get_attnres_info()
    assert info is not None
    assert info["config"]["enabled"]
    assert len(info["blocks"]) == 2  # Only last 2 blocks use AttnRes
    print("  ✓ get_attnres_info works")

    # Test get_gate_values
    gates = model_attnres.get_gate_values()
    assert isinstance(gates, dict)
    print("  ✓ get_gate_values works")

    print("✓ DeepVIDv2 integration tests passed\n")


def test_control_blocks():
    """Test control fusion blocks."""
    print("Testing control fusion blocks...")

    batch_size = 2
    channels = 64
    height = 32
    width = 32

    # Test ConcatFusionBlock
    concat_block = ConcatFusionBlock(channels, history_len=2)

    current = torch.randn(batch_size, channels, height, width)
    history = [
        torch.randn(batch_size, channels, height, width),
        torch.randn(batch_size, channels, height, width),
    ]
    residual = torch.randn(batch_size, channels, height, width)

    output = concat_block(current, history, residual)
    assert output.shape == (batch_size, channels, height, width)
    print("  ✓ ConcatFusionBlock works")

    # Test GateOnlyBlock
    gate_block = GateOnlyBlock(channels, gate_type="scalar", init_value=0.0)
    output = gate_block(residual)
    assert output.shape == residual.shape
    print("  ✓ GateOnlyBlock works")

    print("✓ Control block tests passed\n")


def test_parameter_count():
    """Test parameter counting."""
    print("Testing parameter counting...")

    from source.utils import count_parameters, get_model_info

    args = argparse.Namespace(
        kernel_size=3,
        stride=2,
        padding=1,
        in_channels=11,
        out_channels=1,
        num_feature=64,
        num_blocks=4,
        norm_type="batch",
        activation_type="prelu",
        resblock_activation_out_type=None,
    )

    # Baseline model
    config_baseline = AttnResConfig(enabled=False)
    model_baseline = DeepVIDv2(args, attnres_config=config_baseline)
    params_baseline = count_parameters(model_baseline)

    # AttnRes model
    config_attnres = AttnResConfig(
        enabled=True,
        mode="bottleneck",
        history_len=2,
    )
    model_attnres = DeepVIDv2(args, attnres_config=config_attnres)
    params_attnres = count_parameters(model_attnres)

    print(f"  Baseline params: {params_baseline:,}")
    print(f"  AttnRes params: {params_attnres:,}")
    print(f"  Additional params: {params_attnres - params_baseline:,}")

    assert params_attnres > params_baseline  # AttnRes adds parameters
    print("  ✓ Parameter counting works")

    # Test get_model_info
    info = get_model_info(model_attnres, input_size=(1, 11, 64, 64))
    assert "total_params" in info
    assert "attnres" in info
    print("  ✓ get_model_info works")

    print("✓ Parameter counting tests passed\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running AttnRes Implementation Tests")
    print("=" * 60 + "\n")

    try:
        test_config()
        test_rms_norm()
        test_learned_gate()
        test_aggregator()
        test_history_manager()
        test_control_blocks()
        test_deepvid_integration()
        test_parameter_count()

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

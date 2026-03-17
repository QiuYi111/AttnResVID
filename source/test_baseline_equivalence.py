"""
Test baseline equivalence - verify AttnRes disabled = same as original DeepVIDv2.

This test ensures that when AttnRes is disabled, the model produces
exactly the same outputs as the original DeepVIDv2 without any AttnRes code.
"""

import sys
import os

# Add parent directory to path for 'source' module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from argparse import Namespace

# Import the model and config
from source.network_collection import DeepVIDv2
from source.attnres.config import AttnResConfig


def create_default_args():
    """Create default args for DeepVIDv2."""
    return Namespace(
        kernel_size=3,
        stride=1,
        padding=1,
        in_channels=1,
        out_channels=1,
        num_feature=32,
        num_blocks=4,
        norm_type="batch",
        activation_type="relu",
        resblock_activation_out_type="relu",
    )


def test_baseline_equivalence():
    """Test that AttnRes disabled produces identical results to no AttnRes at all."""
    print("=" * 60)
    print("Testing Baseline Equivalence")
    print("=" * 60)

    # Create args
    args = create_default_args()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create model without AttnRes (baseline)
    model_baseline = DeepVIDv2(args, attnres_config=None)

    # Reset seed to ensure same initialization
    torch.manual_seed(42)

    # Create model with AttnRes disabled
    attnres_config = AttnResConfig(enabled=False)
    model_disabled = DeepVIDv2(args, attnres_config=attnres_config)

    # Create model with AttnRes enabled in "off" mode
    torch.manual_seed(42)
    attnres_config_off = AttnResConfig(enabled=True, mode="off")
    model_off_mode = DeepVIDv2(args, attnres_config=attnres_config_off)

    # Create random input
    torch.manual_seed(42)
    x = torch.randn(2, 1, 64, 64)

    # Forward pass
    with torch.no_grad():
        out_baseline = model_baseline(x)
        out_disabled = model_disabled(x)
        out_off_mode = model_off_mode(x)

    # Check numerical equivalence
    diff_baseline_disabled = torch.max(torch.abs(out_baseline - out_disabled)).item()
    diff_baseline_off = torch.max(torch.abs(out_baseline - out_off_mode)).item()

    print(f"\nMax difference (baseline vs disabled): {diff_baseline_disabled:.2e}")
    print(f"Max difference (baseline vs off mode): {diff_baseline_off:.2e}")

    # Assert numerical equivalence (allowing for small floating point errors)
    tolerance = 1e-6
    assert diff_baseline_disabled < tolerance, \
        f"Baseline and disabled AttnRes should be identical (diff={diff_baseline_disabled:.2e})"
    assert diff_baseline_off < tolerance, \
        f"Baseline and off mode should be identical (diff={diff_baseline_off:.2e})"

    print("\n✓ Baseline equivalence test PASSED")
    print("  - AttnRes disabled produces identical results to baseline")
    print("  - AttnRes 'off' mode produces identical results to baseline")

    return True


def test_parameter_count():
    """Test that AttnRes disabled adds no extra parameters."""
    print("\n" + "=" * 60)
    print("Testing Parameter Count")
    print("=" * 60)

    args = create_default_args()

    # Create models
    model_baseline = DeepVIDv2(args, attnres_config=None)
    model_disabled = DeepVIDv2(args, attnres_config=AttnResConfig(enabled=False))

    # Count parameters
    params_baseline = sum(p.numel() for p in model_baseline.parameters())
    params_disabled = sum(p.numel() for p in model_disabled.parameters())

    print(f"\nBaseline parameters: {params_baseline:,}")
    print(f"Disabled parameters: {params_disabled:,}")
    print(f"Difference: {params_disabled - params_baseline:,}")

    assert params_baseline == params_disabled, \
        f"AttnRes disabled should add no parameters (baseline={params_baseline}, disabled={params_disabled})"

    print("\n✓ Parameter count test PASSED")
    print("  - AttnRes disabled adds zero extra parameters")

    return True


def test_attnres_increases_params():
    """Test that AttnRes enabled actually increases parameter count."""
    print("\n" + "=" * 60)
    print("Testing AttnRes Parameter Increase")
    print("=" * 60)

    args = create_default_args()

    # Create models
    model_baseline = DeepVIDv2(args, attnres_config=None)
    model_attnres = DeepVIDv2(args, attnres_config=AttnResConfig(
        enabled=True,
        mode="bottleneck",
        history_len=2,
    ))

    # Count parameters
    params_baseline = sum(p.numel() for p in model_baseline.parameters())
    params_attnres = sum(p.numel() for p in model_attnres.parameters())

    print(f"\nBaseline parameters: {params_baseline:,}")
    print(f"AttnRes parameters: {params_attnres:,}")
    print(f"Increase: {params_attnres - params_baseline:,}")
    print(f"Percentage: {100 * (params_attnres - params_baseline) / params_baseline:.2f}%")

    assert params_attnres > params_baseline, \
        "AttnRes enabled should increase parameter count"

    print("\n✓ AttnRes parameter increase test PASSED")
    print("  - AttnRes enabled adds parameters as expected")

    return True


def test_forward_shapes():
    """Test that all modes produce correct output shapes."""
    print("\n" + "=" * 60)
    print("Testing Output Shapes")
    print("=" * 60)

    args = create_default_args()
    modes = ["off", "bottleneck", "stagewise"]
    batch_size = 2
    height, width = 64, 64

    for mode in modes:
        config = AttnResConfig(enabled=True, mode=mode) if mode != "baseline" else None
        model = DeepVIDv2(args, attnres_config=config)

        x = torch.randn(batch_size, args.in_channels, height, width)
        with torch.no_grad():
            out = model(x)

        expected_shape = (batch_size, args.out_channels, height, width)
        assert out.shape == expected_shape, \
            f"Mode {mode} produced shape {out.shape}, expected {expected_shape}"

        print(f"  {mode:12s}: {out.shape}")

    print("\n✓ Output shape test PASSED")
    print("  - All modes produce correct output shapes")

    return True


def run_all_tests():
    """Run all baseline equivalence tests."""
    print("\n" + "=" * 60)
    print("BASELINE EQUIVALENCE TEST SUITE")
    print("=" * 60)

    tests = [
        ("Baseline Equivalence", test_baseline_equivalence),
        ("Parameter Count", test_parameter_count),
        ("AttnRes Parameter Increase", test_attnres_increases_params),
        ("Output Shapes", test_forward_shapes),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

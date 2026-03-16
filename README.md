# AttnResVID: Attention-Residual Mechanisms for Low-Photon Voltage Imaging Denoising

An extension of [DeepVIDv2](https://github.com/bu-cisl/DeepVIDv2) with learnable depth-wise feature aggregation for improved low-photon voltage imaging denoising.

> **Paper coming soon.** This repository contains the implementation for our ongoing research on attention-based residual learning in self-supervised video denoising.

---

## Overview

AttnResVID introduces **Attention-Residual (AttnRes)** mechanisms into DeepVIDv2, enabling the network to adaptively aggregate historical features across network depth. Unlike standard fixed residual connections (`y = x + F(x)`), AttnRes learns to weight contributions from previous layers:

```
y = x + F(x) + γ · A(x_{l-K}, ..., x_{l-1}; x_l)
```

where `A(·)` is a depth-wise attention aggregator and `γ` is a learnable gate.

### Key Features

- **Minimal Intrusion**: Wraps existing ResBlocks without rewriting the core architecture
- **Configurable**: All parameters controlled via CLI arguments
- **Ablation-Friendly**: Multiple fusion modes for controlled experiments
- **Training Stable**: Gated injection with learnable initialization

---

## Installation

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/QiuYi111/AttnResVID.git
cd AttnResVID

# Install dependencies
uv sync
```

### Using conda

```bash
# Create conda environment
conda env create -f environment.yml
conda activate attnresvid

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

## Quick Start

### Baseline Training (DeepVIDv2)

```bash
uv run python scripts/train.py \
  --noisy-data-paths ./datasets \
  --model-string baseline_run
```

### With Attention-Residual (Bottleneck Mode)

```bash
uv run python scripts/train.py \
  --noisy-data-paths ./datasets \
  --attnres-enabled \
  --attnres-mode bottleneck \
  --attnres-history-len 2 \
  --model-string attnres_run
```

### Inference

```bash
uv run python scripts/inference.py \
  --noisy-data-paths ./datasets \
  --model attnres_run
```

---

## Experiments

### Main Experiments

| Experiment | Description | Command |
|------------|-------------|---------|
| **E0** | Baseline DeepVIDv2 | `--attnres-enabled false` |
| **E1** | Bottleneck-only (last 2 blocks) | `--attnres-enabled --attnres-mode bottleneck --attnres-history-len 2` |
| **E2** | Full bottleneck (history_len=3) | `--attnres-enabled --attnres-history-len 3` |
| **E3** | Bottleneck + Decoder | `--attnres-enabled --attnres-mode bottleneck_decoder` |
| **E4** | Stage-wise injection | `--attnres-enabled --attnres-mode stagewise --attnres-history-len 4` |

### Control Experiments

| Experiment | Description | Command |
|------------|-------------|---------|
| **C1** | Concat fusion (no attention) | `--attnres-enabled --attnres-fusion-mode concat` |
| **C2** | Gate-only (no history) | `--attnres-enabled --attnres-fusion-mode gate_only` |

### Parameter Sweeps

```bash
# Temperature sweep
uv run python scripts/train.py --attnres-enabled --attnres-temperature 0.5 --model-string temp_0.5
uv run python scripts/train.py --attnres-enabled --attnres-temperature 1.0 --model-string temp_1.0
uv run python scripts/train.py --attnres-enabled --attnres-temperature 2.0 --model-string temp_2.0

# History length sweep
uv run python scripts/train.py --attnres-enabled --attnres-history-len 2 --model-string hist_2
uv run python scripts/train.py --attnres-enabled --attnres-history-len 3 --model-string hist_3
uv run python scripts/train.py --attnres-enabled --attnres-history-len 4 --model-string hist_4
```

---

## Configuration

### AttnRes Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--attnres-enabled` | flag | False | Enable AttnRes mechanism |
| `--attnres-mode` | str | bottleneck | Mode: off, bottleneck, bottleneck_decoder, stagewise |
| `--attnres-history-len` | int | 2 | Number of historical features to aggregate |
| `--attnres-temperature` | float | 1.0 | Softmax temperature (lower = sharper) |
| `--attnres-gate-init` | float | 0.0 | Initial gate value (0 = start as identity) |
| `--attnres-score-fn` | str | gap_linear | Score function: gap_linear, conv1x1_gap_linear |
| `--attnres-gate-type` | str | scalar | Gate type: scalar, channel |
| `--attnres-detach-history` | flag | False | Detach history for stability |
| `--attnres-fusion-mode` | str | attention | Fusion: attention, concat, gate_only |
| `--attnres-share-proj` | flag | False | Share projections across blocks |

### Example Configurations

```bash
# Conservative: minimal injection, high stability
--attnres-enabled --attnres-mode bottleneck --attnres-history-len 2 --attnres-gate-init 0.0 --attnres-temperature 2.0

# Aggressive: full history, low temperature
--attnres-enabled --attnres-mode stagewise --attnres-history-len 4 --attnres-temperature 0.5

# Experimental: channel-wise gating
--attnres-enabled --attnres-gate-type channel --attnres-fusion-mode attention
```

---

## Architecture

### DeepVIDv2 with AttnRes

```
Input Block (ConvBlock)
    ↓
┌─────────────────────────────────────────┐
│  4 × ResBlock (with optional AttnRes)    │
│  ┌─────────────────────────────────────┐ │
│  │  x → ResBlock → y                    │ │
│  │                    ↓                 │ │
│  │  + AttnRes: A(history) → gated      │ │
│  │                    ↓                 │ │
│  │  output = x + residual + gated_attn  │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
    ↓
Pre-out Block (ConvBlock)
    ↓
Output Block (2 × ConvBlock)
```

### AttnRes Components

- **`AttnResConfig`**: Configuration dataclass with all hyperparameters
- **`DepthAttentionAggregator`**: GAP → learned query → softmax attention
- **`LearnedGate`**: Scalar or channel-wise gating with sigmoid activation
- **`StageHistoryManager`**: Stage-local feature history tracking
- **`AttnResBlockWrapper`**: Non-invasive wrapper for ResBlocks

---

## Module Structure

```
source/attnres/
├── __init__.py          # Module exports
├── config.py            # AttnResConfig dataclass
├── aggregator.py        # DepthAttentionAggregator, RMSNorm2d, LearnedGate
├── gated_residual.py    # GatedAttnResidual
├── history_manager.py   # StageHistoryManager
├── wrapper.py           # AttnResBlockWrapper
└── control.py           # Control variants (ConcatFusionBlock, GateOnlyBlock)
```

---

## Testing

Run the test suite to verify installation:

```bash
uv run python test_attnres.py
```

Expected output:
```
============================================================
✓ ALL TESTS PASSED!
============================================================
```

---

## Citation

```bibtex
@article{attnresvid2025,
  title={Attention-Residual Mechanisms for Low-Photon Voltage Imaging Denoising},
  author={[Coming Soon]},
  journal={},
  year={2025}
}
```

Please also cite the original DeepVIDv2 paper:

```bibtex
@article{liu2024deepvid2,
  title={DeepVID v2: Self-Supervised Denoising with Decoupled Spatiotemporal Enhancement for Low-Photon Voltage Imaging},
  author={Liu, Chang and Lu, Jiaming and Wu, Yuting and Ye, Xinhao and Ahrens, Adam M and Platisa, Jelena and Tian, Lu},
  journal={bioRxiv},
  pages={2024--05},
  year={2024},
  publisher={Cold Spring Harbor Laboratory Preprint}
}
```

---

## Acknowledgments

This work extends [DeepVIDv2](https://github.com/bu-cisl/DeepVIDv2) by the Boston University Computational Imaging & Synthetic Aperture Systems Lab.

---

## License

This project inherits the license from the original DeepVIDv2 repository.

---

## Contact

For questions about AttnResVID, please open an issue on GitHub or contact [Authors].

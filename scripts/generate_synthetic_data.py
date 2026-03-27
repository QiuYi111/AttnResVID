"""
Generate synthetic calcium imaging data for testing AttnResVID.

This script creates simulated voltage imaging data with:
- Time-varying baseline (simulating photobleaching)
- Spatiotemporal correlated signal (simulating neuronal activity)
- Poisson-Gaussian noise (low-photon imaging characteristics)
"""

import numpy as np
import os
from pathlib import Path
from skimage import io
from typing import Tuple


def generate_synthetic_data(
    output_path: str,
    num_frames: int = 500,
    height: int = 256,
    width: int = 256,
    num_neurons: int = 20,
    snr_db: float = 10.0,
    trend_order: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic calcium imaging data.

    Args:
        output_path: Path to save the data
        num_frames: Number of frames in the video
        height: Frame height
        width: Frame width
        num_neurons: Number of simulated neurons
        snr_db: Signal-to-noise ratio in dB
        trend_order: Polynomial order for baseline drift
        seed: Random seed

    Returns:
        clean_data: Noise-free synthetic data
        noisy_data: Data with added noise
    """
    np.random.seed(seed)

    print(f"Generating synthetic calcium imaging data:")
    print(f"  Frames: {num_frames}")
    print(f"  Size: {height}x{width}")
    print(f"  Neurons: {num_neurons}")
    print(f"  SNR: {snr_db} dB")

    # Initialize clean data
    clean_data = np.zeros((num_frames, height, width), dtype=np.float32)

    # Generate spatially localized neuron patterns
    neuron_centers = []
    for _ in range(num_neurons):
        cy = np.random.randint(20, height - 20)
        cx = np.random.randint(20, width - 20)
        radius = np.random.randint(8, 20)
        neuron_centers.append((cy, cx, radius))

    # Generate temporal signals for each neuron
    # Using calcium response dynamics: exponential rise and decay
    t = np.arange(num_frames)
    temporal_signals = []

    for _ in range(num_neurons):
        # Random spike times
        num_spikes = np.random.randint(5, 15)
        spike_times = np.sort(np.random.choice(num_frames, num_spikes, replace=False))

        # Generate calcium signal
        signal = np.zeros(num_frames)
        tau_rise = 2.0  # frames
        tau_decay = 15.0  # frames

        for spike_time in spike_times:
            # Double exponential for calcium dynamics
            dt = t - spike_time
            spike_signal = np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)
            spike_signal[dt < 0] = 0
            signal += spike_signal

        # Normalize
        signal = signal / (np.max(signal) + 1e-6)
        temporal_signals.append(signal)

    # Spatial component: 2D Gaussian for each neuron
    y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    for frame_idx in range(num_frames):
        frame = np.zeros((height, width))

        for (cy, cx, radius), signal in zip(neuron_centers, temporal_signals):
            # 2D Gaussian spatial pattern
            sigma = radius / 2.0
            gaussian = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))
            frame += signal[frame_idx] * gaussian

        # Add low-frequency background activity
        background = 0.1 * np.sin(2 * np.pi * frame_idx / num_frames) * \
                    np.exp(-((x_grid - width//2)**2 + (y_grid - height//2)**2) / (2 * (width/3)**2))
        frame += background

        clean_data[frame_idx] = frame

    # Add temporal trend (simulating photobleaching)
    trend_coeffs = np.polyfit(np.arange(num_frames), np.mean(clean_data, axis=(1, 2)), trend_order)
    trend = np.polyval(trend_coeffs, np.arange(num_frames))
    trend = trend - np.mean(trend)
    trend = trend.reshape(-1, 1, 1)
    clean_data = clean_data + trend * 0.5

    # Normalize clean data
    clean_data = (clean_data - np.mean(clean_data)) / (np.std(clean_data) + 1e-6)

    # Scale to realistic photon counts
    # Typical voltage imaging: 100-500 photons per pixel
    baseline_photons = 200.0
    signal_amplitude = 100.0
    clean_data_scaled = baseline_photons + clean_data * signal_amplitude
    clean_data_scaled = np.clip(clean_data_scaled, 0, None)

    # Add Poisson-Gaussian noise
    # 1. Poisson noise (shot noise from photon counting)
    noisy_data = np.random.poisson(clean_data_scaled).astype(np.float32)

    # 2. Gaussian noise (readout noise)
    readout_noise_std = 5.0
    noisy_data += np.random.normal(0, readout_noise_std, noisy_data.shape).astype(np.float32)

    # Ensure non-negative
    noisy_data = np.clip(noisy_data, 0, None)

    print(f"  Clean data range: [{clean_data_scaled.min():.1f}, {clean_data_scaled.max():.1f}]")
    print(f"  Noisy data range: [{noisy_data.min():.1f}, {noisy_data.max():.1f}]")

    # Save as multi-page TIFF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    io.imwrite(output_path, noisy_data)
    print(f"  Saved to: {output_path}")

    return clean_data_scaled, noisy_data


def generate_test_dataset(output_dir: str, num_samples: int = 3):
    """Generate multiple test samples."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        filename = output_path / f"synthetic_{i:03d}.tif"
        generate_synthetic_data(
            str(filename),
            num_frames=500,
            height=256,
            width=256,
            num_neurons=20,
            snr_db=10.0,
            seed=42 + i,
        )
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic calcium imaging data")
    parser.add_argument("--output-dir", type=str, default="./datasets/synthetic",
                        help="Output directory for synthetic data")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--frames", type=int, default=500,
                        help="Number of frames per sample")
    parser.add_argument("--height", type=int, default=256,
                        help="Frame height")
    parser.add_argument("--width", type=int, default=256,
                        help="Frame width")
    parser.add_argument("--neurons", type=int, default=20,
                        help="Number of simulated neurons")
    parser.add_argument("--snr", type=float, default=10.0,
                        help="Signal-to-noise ratio in dB")

    args = parser.parse_args()

    generate_test_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )

#!/usr/bin/env python
"""
Interactive browser for OnePhoton calcium imaging data in NWB files.

Uses PyNWB (official NWB library) with lazy loading.

Usage:
    python preview_nwb.py <nwb_file>

Controls:
    - Arrow keys: Navigate frames
    - Space: Play/Pause
    - +/-: Adjust playback speed
    - c: Cycle contrast modes
    - g: Go to frame
    - q: Quit
"""

import pynwb
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable


class NWBBrowser:
    def __init__(self, nwb_path):
        self.nwb_path = nwb_path
        self.io = None
        self.nwb = None
        self.one_photon = None
        self.metadata = {}
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 10  # fps
        self.vmin_default = None
        self.vmax_default = None

        self.open_nwb()
        self.setup_figure()

    def open_nwb(self):
        """Open NWB file using PyNWB."""
        print(f"Opening: {self.nwb_path}")
        self.io = pynwb.NWBHDF5IO(self.nwb_path, 'r')
        self.nwb = self.io.read()

        # Get OnePhotonSeries
        self.one_photon = self.nwb.acquisition['OnePhotonSeries']

        # Metadata
        self.metadata['num_frames'] = self.one_photon.data.shape[0]
        self.metadata['height'] = self.one_photon.data.shape[1]
        self.metadata['width'] = self.one_photon.data.shape[2]
        self.metadata['dtype'] = self.one_photon.data.dtype
        self.metadata['rate'] = self.one_photon.rate

        print(f"\n{'='*50}")
        print(f"Calcium Imaging Data (PyNWB)")
        print(f"{'='*50}")
        print(f"  Session: {self.nwb.session_description}")
        print(f"  Subject: {self.nwb.subject.subject_id} ({self.nwb.subject.species})")
        print(f"  Frames: {self.metadata['num_frames']}")
        print(f"  Size: {self.metadata['height']} x {self.metadata['width']}")
        print(f"  Rate: {self.metadata['rate']} Hz")
        print(f"  Duration: {self.metadata['num_frames'] / self.metadata['rate']:.1f} sec")
        print(f"{'='*50}\n")

        # Sample for contrast
        print("Sampling frames for contrast...")
        sample_indices = np.linspace(0, self.metadata['num_frames'] - 1,
                                     min(50, self.metadata['num_frames']), dtype=int)
        samples = []
        for idx in sample_indices:
            frame = self.one_photon.data[idx]
            samples.append(frame.flatten())

        all_samples = np.concatenate(samples)
        self.vmin_default = np.percentile(all_samples, 1)
        self.vmax_default = np.percentile(all_samples, 99)
        print(f"Contrast range: [{self.vmin_default:.1f}, {self.vmax_default:.1f}]")

    def get_frame(self, idx):
        """Get a single frame (lazy loaded)."""
        return self.one_photon.data[idx]

    def setup_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title(f"NWB Browser - {self.nwb_path}")

        # Initial frame
        frame = self.get_frame(0)
        self.img = self.ax.imshow(frame, cmap='gray',
                                   vmin=self.vmin_default, vmax=self.vmax_default)

        self.ax.set_title(f"Frame 0 / {self.metadata['num_frames'] - 1}")
        self.ax.axis('off')

        # Colorbar
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.fig.colorbar(self.img, cax=cax)

        # Info text
        self.info_text = self.fig.text(0.02, 0.02, "", fontsize=10, family='monospace')
        self.update_info()

        # Event handlers
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.anim = FuncAnimation(self.fig, self.update, interval=1000//self.playback_speed,
                                  blit=False, cache_frame_data=False)

    def update_info(self):
        frame = self.get_frame(self.current_frame)
        info = [
            f"Frame: {self.current_frame} / {self.metadata['num_frames'] - 1}",
            f"Time: {self.current_frame / self.metadata['rate']:.2f} s",
            f"Min: {frame.min():.0f}  Max: {frame.max():.0f}  Mean: {frame.mean():.0f}",
            f"Speed: {self.playback_speed} fps | Playing: {self.is_playing}",
        ]
        self.info_text.set_text("\n".join(info))

    def update(self, frame_num):
        if self.is_playing:
            self.current_frame = (self.current_frame + 1) % self.metadata['num_frames']
            self.show_frame()
        return [self.img]

    def show_frame(self):
        frame = self.get_frame(self.current_frame)
        self.img.set_data(frame)
        time_sec = self.current_frame / self.metadata['rate']
        self.ax.set_title(f"Frame {self.current_frame} / {self.metadata['num_frames'] - 1} | {time_sec:.2f} s")
        self.update_info()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'q':
            plt.close(self.fig)
            return

        elif event.key == 'right':
            self.current_frame = min(self.current_frame + 1, self.metadata['num_frames'] - 1)
            self.is_playing = False
            self.show_frame()

        elif event.key == 'left':
            self.current_frame = max(self.current_frame - 1, 0)
            self.is_playing = False
            self.show_frame()

        elif event.key == ' ':
            self.is_playing = not self.is_playing

        elif event.key in ['+', '=']:
            self.playback_speed = min(self.playback_speed * 2, 60)
            self.anim.event_source.interval = 1000 // self.playback_speed
            self.update_info()

        elif event.key == '-':
            self.playback_speed = max(self.playback_speed // 2, 1)
            self.anim.event_source.interval = 1000 // self.playback_speed
            self.update_info()

        elif event.key == 'r':
            self.current_frame = 0
            self.is_playing = False
            self.show_frame()

        elif event.key == 'c':
            # Cycle contrast modes
            if self.img.get_clim() == (self.vmin_default, self.vmax_default):
                # Current frame range
                frame = self.get_frame(self.current_frame)
                self.img.set_clim(frame.min(), frame.max())
            else:
                # Back to auto
                self.img.set_clim(self.vmin_default, self.vmax_default)
            self.update_info()

        elif event.key == 'home':
            self.current_frame = 0
            self.is_playing = False
            self.show_frame()

        elif event.key == 'end':
            self.current_frame = self.metadata['num_frames'] - 1
            self.is_playing = False
            self.show_frame()

    def show(self):
        print("\nControls:")
        print("  ←/→ : Navigate frames")
        print("  Space: Play/Pause")
        print("  +/-  : Adjust speed")
        print("  c    : Toggle contrast")
        print("  r    : Reset to frame 0")
        print("  q    : Quit\n")
        plt.show()

    def close(self):
        if self.io is not None:
            self.io.close()


def main():
    parser = argparse.ArgumentParser(description="Browse NWB calcium imaging data")
    parser.add_argument("nwb_file", help="Path to NWB file")
    args = parser.parse_args()

    browser = NWBBrowser(args.nwb_file)
    try:
        browser.show()
    finally:
        browser.close()


if __name__ == "__main__":
    main()

"""
NWB Dataset loader for DeepVIDv2 using PyNWB (official library)

Uses lazy loading via PyNWB - suitable for large NWB files.
"""

import numpy as np
import torch
import pynwb
from torch.utils.data import Dataset

import source.dataset_collection as base_ds


class NWBVIDataset(base_ds.VIDataset):
    """
    Load calcium imaging data from NWB files using PyNWB.

    Uses lazy loading - frames are only loaded when accessed.
    """

    def __init__(self, args):
        self.io = None
        self.nwb = None
        self.one_photon = None
        super(NWBVIDataset, self).__init__()

    def parse_args(self, args):
        self.nwb_path = args.nwb_path
        self.data_path = args.nwb_path  # Compatibility

    def imread(self):
        """Open NWB file and setup lazy loading."""
        print(f"Opening NWB file: {self.nwb_path}")
        self.io = pynwb.NWBHDF5IO(self.nwb_path, 'r')
        self.nwb = self.io.read()
        self.one_photon = self.nwb.acquisition['OnePhotonSeries']

        self.num_frames = self.one_photon.data.shape[0]
        self.img_rows = self.one_photon.data.shape[1]
        self.img_cols = self.one_photon.data.shape[2]

        print(f"  Frames: {self.num_frames}, Size: {self.img_rows}x{self.img_cols}")

        # Compute statistics via sampling for detrending/normalization
        print("Computing statistics from sampled frames...")
        self._compute_statistics()

    def _compute_statistics(self):
        """Compute detrending trend and normalization parameters from samples."""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from scipy.interpolate import interp1d

        # Sample frames for efficient computation
        n_samples = min(1000, self.num_frames)
        sample_stride = max(1, self.num_frames // n_samples)
        sample_indices = np.arange(0, self.num_frames, sample_stride)

        # Sample means for detrending
        trace_samples = []
        for idx in sample_indices:
            frame = self.one_photon.data[idx].astype('float32')
            trace_samples.append(np.mean(frame))

        # Interpolate to full length
        X = np.arange(self.num_frames)
        trace_full = interp1d(sample_indices, trace_samples, kind='linear',
                             fill_value='extrapolate')(X)

        # Polynomial trend
        X_sample = sample_indices.reshape(-1, 1)
        pf = PolynomialFeatures(2)
        Xp = pf.fit_transform(X_sample)
        md = LinearRegression()
        md.fit(Xp, trace_samples)

        X_full = X.reshape(-1, 1)
        Xp_full = pf.fit_transform(X_full)
        self.trend = md.predict(Xp_full).astype('float32')

        # Compute mean/std from samples
        mean_samples = []
        std_samples = []
        for idx in sample_indices:
            frame = self.one_photon.data[idx].astype('float32')
            detrended = frame - self.trend[idx]
            mean_samples.append(np.mean(detrended))
            std_samples.append(np.mean((detrended - np.mean(detrended))**2))

        self.local_mean = np.mean(mean_samples)
        self.local_std = np.sqrt(np.mean(std_samples))

        print(f"  Normalization: mean={self.local_mean:.2f}, std={self.local_std:.2f}")

    def get_frame(self, idx):
        """Get a preprocessed frame (lazy loaded)."""
        frame = self.one_photon.data[idx].astype('float32')
        frame = frame - self.trend[idx]
        frame = (frame - self.local_mean) / self.local_std
        return frame

    def preprocess(self):
        """Preprocessing handled on-the-fly in get_frame."""
        pass

    def __del__(self):
        """Close NWB file."""
        if self.io is not None:
            self.io.close()


class NWBDeepVIDv2Dataset(NWBVIDataset, base_ds.DeepVIDv2Dataset):
    """
    NWB dataset for DeepVIDv2 with edge prior.

    Uses PyNWB with lazy loading.
    """

    def __init__(self, args):
        self.io = None
        self.nwb = None
        self.one_photon = None

        # Initialize with base VIDataset
        base_ds.VIDataset.__init__(self, args)

    def parse_args(self, args):
        self.nwb_path = args.nwb_path
        self.data_path = args.nwb_path

        # DIPDataset params
        self.pre_frame = args.input_pre_post_frame
        self.post_frame = args.input_pre_post_frame
        self.pre_post_omission = args.pre_post_omission

        # N2VDataset params
        self.blind_pixel_ratio = args.blind_pixel_ratio
        self.blind_pixel_method = args.blind_pixel_method

        # DeepVIDv2 params
        self.edge_gaussian_size = args.edge_gaussian_size
        self.edge_gaussian_sigma = args.edge_gaussian_sigma
        self.edge_pre_post_frame = args.edge_pre_post_frame
        self.edge_source_frame = args.edge_source_frame

    def imread(self):
        """Use NWBVIDataset's imread."""
        NWBVIDataset.imread(self)

    def preprocess(self):
        """Setup edge prior after base preprocessing."""
        from torchvision.transforms import GaussianBlur
        import source.network_collection as nc

        self.gaussian = GaussianBlur(self.edge_gaussian_size, self.edge_gaussian_sigma)
        self.sobel = nc.SobelBlock()
        self.sobel = self.sobel.to(self.device)

        # Compute edge prior from samples
        print("Computing edge prior...")
        n_samples = min(100, self.num_frames)
        sample_stride = max(1, self.num_frames // n_samples)

        mean_frames = []
        for idx in range(0, self.num_frames, sample_stride):
            mean_frames.append(self.get_frame(idx))

        self.mean_frame = np.mean(mean_frames, axis=0)
        self.std_frame = np.std(mean_frames, axis=0)

        if self.edge_source_frame == "mean":
            self.edge_prior_global = self.extract_edge(self.mean_frame)
        elif self.edge_source_frame == "std":
            self.edge_prior_global = self.extract_edge(self.std_frame)

    def __getitem__(self, index):
        """
        Load frames on-demand with edge prior.

        Returns:
            input: (pre_frame + post_frame + 1 + 4, H, W)
            output: (2, H, W) - frame + mask
        """
        index_frame = self.list_samples[index]

        # Input indices (context frames)
        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        sel = (input_index >= index_frame - self.pre_post_omission) & (
            input_index <= index_frame + self.pre_post_omission
        )
        input_index = input_index[~sel]

        # Load contextual frames
        frames = [self.get_frame(idx) for idx in input_index]
        context_frames = np.stack(frames, axis=0)

        # Load central frame
        data_img_central_frame = self.get_frame(index_frame)

        # Apply blind spot if needed
        if self.blind_pixel_ratio > 0:
            data_img_modified, mask = self._generate_blind_image(data_img_central_frame)
        else:
            data_img_modified = data_img_central_frame
            mask = np.zeros_like(data_img_central_frame)

        # Combine input
        data_img_input = np.concatenate(
            (context_frames, data_img_modified[np.newaxis, :, :]),
            axis=0
        )

        # Output
        data_img_output = np.concatenate(
            (data_img_central_frame[np.newaxis, :, :],
             mask[np.newaxis, :, :]),
            axis=0
        )

        input = torch.from_numpy(data_img_input.astype("float32"))
        output = torch.from_numpy(data_img_output.astype("float32"))

        # Add edge prior
        if self.edge_pre_post_frame is None:
            edge_prior = torch.from_numpy(self.edge_prior_global)
        else:
            edge_index = np.arange(
                max(0, index_frame - self.edge_pre_post_frame),
                min(self.num_frames, index_frame + self.edge_pre_post_frame + 1)
            )

            if self.edge_source_frame == "mean":
                local_mean = np.mean([self.get_frame(i) for i in edge_index], axis=0)
                edge_prior = self.extract_edge(local_mean)
            else:
                local_std = np.std([self.get_frame(i) for i in edge_index], axis=0)
                edge_prior = self.extract_edge(local_std)

            edge_prior = torch.from_numpy(edge_prior)

        input = torch.cat([input, edge_prior], dim=0)
        return input, output

    def _generate_blind_image(self, img):
        """Generate blind-spot image for Noise2Void."""
        img_rows, img_cols = img.shape

        modified = np.copy(img)
        mask = np.zeros(img.shape)

        n_pix = int(self.blind_pixel_ratio * img_rows * img_cols)
        idx = np.random.permutation(img_rows * img_cols)[:n_pix]
        rows, cols = np.unravel_index(idx, (img_rows, img_cols))

        if self.blind_pixel_method == "zeros":
            modified[rows, cols] = 0
        elif self.blind_pixel_method == "replace":
            src = np.random.permutation(img_rows * img_cols)[:n_pix]
            src_rows, src_cols = np.unravel_index(src, (img_rows, img_cols))
            modified[rows, cols] = img[src_rows, src_cols]

        mask[rows, cols] = 1
        return modified, mask


def test_nwb_loading(nwb_path: str):
    """Test function to verify NWB loading works."""
    import pynwb

    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()

    ops = nwb.acquisition['OnePhotonSeries']

    print(f"Session: {nwb.session_description}")
    print(f"Data shape: {ops.data.shape}")
    print(f"Rate: {ops.rate} Hz")

    # Test lazy loading
    print(f"Frame 0: min={ops.data[0].min()}, max={ops.data[0].max()}")
    print(f"Frame 100: min={ops.data[100].min()}, max={ops.data[100].max()}")

    io.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb", type=str, help="Test NWB file")
    args = parser.parse_args()

    if args.nwb:
        test_nwb_loading(args.nwb)

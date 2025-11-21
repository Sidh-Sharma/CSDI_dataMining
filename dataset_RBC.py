"""
dataset_RBC.py

PyTorch Dataset for Rayleigh–Bénard HDF5 files in `data/fluid_rb`.

Behavior:
- Reads HDF5 files containing 5 simulation runs each.
- For each simulation produces an example shaped (length, features) where
  length = number of timesteps (200) and features = 4 * (H'*W') after
  downsampling spatial grid to `downsample_size` (default 64x64).
- Uses PyTorch's `interpolate` (bicubic) for downsampling.
- Computes and stores mean/std across the training set (per-feature).
- Produces `observed_data`, `observed_mask`, `timepoints`, `gt_mask` in
  the same dictionary format as other datasets in this repo.

Example usage:
    from dataset_RBC import get_dataloader
    train_loader, val_loader, test_loader = get_dataloader(seed=1, nfold=0, batch_size=8)

"""

import os
import glob
import pickle
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class RBC_Dataset(Dataset):
    def __init__(
        self,
        file_list: List[str],
        downsample_size: Tuple[int, int] = (64, 64),
        cache_path: str = None,
        recompute: bool = False,
        mean: np.ndarray = None,
        std: np.ndarray = None,
    ):
        """
        file_list: list of hdf5 file paths
        downsample_size: (H, W) target spatial resolution
        cache_path: optional path to cache processed dataset (pickle)
        recompute: if True, ignore cache and reprocess files
        """
        self.file_list = sorted(file_list)
        self.downsample_size = downsample_size
        self.cache_path = cache_path

        if cache_path is not None and os.path.isfile(cache_path) and not recompute:
            with open(cache_path, "rb") as f:
                (
                    self.observed_values,
                    self.observed_masks,
                    self.gt_masks,
                    self.timepoints,
                    cached_mean,
                    cached_std,
                ) = pickle.load(f)
                # if external mean/std passed, prefer that; else use cached
                self.mean = cached_mean if mean is None else mean
                self.std = cached_std if std is None else std
        else:
            (
                self.observed_values,
                self.observed_masks,
                self.gt_masks,
                self.timepoints,
            ) = self._process_files(self.file_list, self.downsample_size)

            # default masks: ones (no missing) — CSDI handles masking in training
            self.observed_masks = np.ones_like(self.observed_values, dtype=np.float32)
            self.gt_masks = np.ones_like(self.observed_values, dtype=np.float32)
            # compute per-feature mean/std across all samples and time
            data_flat = self.observed_values.reshape(-1, self.observed_values.shape[-1])
            computed_mean = data_flat.mean(axis=0).astype(np.float32)
            computed_std = data_flat.std(axis=0).astype(np.float32)
            # guard against zero std
            computed_std[computed_std == 0] = 1.0

            # if external mean/std provided, use them, otherwise use computed
            self.mean = computed_mean if mean is None else mean.astype(np.float32)
            self.std = computed_std if std is None else std.astype(np.float32)

            if cache_path is not None:
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        [
                            self.observed_values,
                            self.observed_masks,
                            self.gt_masks,
                            self.timepoints,
                            self.mean,
                            self.std,
                        ],
                        f,
                    )

        # store shapes
        self.n_samples = len(self.observed_values)
        self.length = self.observed_values.shape[1]
        self.features = self.observed_values.shape[2]

    def _process_files(self, file_list: List[str], downsample_size: Tuple[int, int]):
        """
        Read HDF5 files and produce arrays:
          observed_values: (N_samples, length, features)
          timepoints: (length,) same for all
        Each file contains 5 simulations; each simulation becomes one sample.
        """
        samples = []
        timepoints = None

        for fp in file_list:
            try:
                with h5py.File(fp, "r") as f:
                    # read needed datasets
                    # buoyancy, pressure under t0_fields; velocity under t1_fields
                    buoy = f["t0_fields"]["buoyancy"][()]  # shape (5, T, H, W)
                    pres = f["t0_fields"]["pressure"][()]
                    vel = f["t1_fields"]["velocity"][()]  # shape (5, T, H, W, 2)

                    # timepoints in 'dimensions/time'
                    if timepoints is None and "dimensions" in f and "time" in f["dimensions"]:
                        timepoints = f["dimensions"]["time"][()].astype(np.float32)

                    # ensure shapes
                    assert buoy.ndim == 4

                    num_sims = buoy.shape[0]
                    T = buoy.shape[1]

                    # build per-simulation arrays
                    for s in range(num_sims):
                        # each channel shape (T,H,W)
                        ch_b = buoy[s]  # (T,H,W)
                        ch_p = pres[s]
                        ch_vx = vel[s][..., 0]
                        ch_vy = vel[s][..., 1]

                        # stack to (C, T, H, W)
                        stacked = np.stack([ch_b, ch_p, ch_vx, ch_vy], axis=0).astype(np.float32)

                        # downsample spatial dims
                        ds = self._downsample_stack(stacked, downsample_size)  # (C, T, h, w)

                        # flatten spatial dims per time -> (T, C*h*w)
                        C, TT, H2, W2 = ds.shape
                        flattened = ds.transpose(1, 0, 2, 3).reshape(TT, C * H2 * W2)

                        samples.append(flattened)

            except Exception as e:
                print(f"Failed reading {fp}: {e}")
                continue

        if timepoints is None:
            # fallback: use a generic range using first sample length
            T = samples[0].shape[0]
            timepoints = np.arange(T, dtype=np.float32)

        observed_values = np.stack(samples, axis=0).astype(np.float32)  # (N, T, features)
        observed_masks = np.ones_like(observed_values, dtype=np.float32)
        gt_masks = np.ones_like(observed_values, dtype=np.float32)

        return observed_values, observed_masks, gt_masks, timepoints

    def _downsample_stack(self, stack: np.ndarray, downsample_size: Tuple[int, int]):
        """
        stack: (C, T, H, W)
        We'll convert to torch tensor shape (T, C, H, W), use F.interpolate.
        Returns (C, T, h, w) as numpy array.
        """
        C, T, H, W = stack.shape
        # convert to (T, C, H, W)
        t = torch.from_numpy(stack.transpose(1, 0, 2, 3)).float()
        # interpolate expects shape (N, C, H, W)
        with torch.no_grad():
            out = F.interpolate(
                t, size=downsample_size, mode="bicubic", align_corners=False
            )
        # out shape (T, C, h, w) -> transpose back to (C, T, h, w)
        out_np = out.numpy().transpose(1, 0, 2, 3)
        return out_np

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return dict matching other dataset format
        data = self.observed_values[idx]  # (T, features)
        mask = self.observed_masks[idx]
        gt = self.gt_masks[idx]
        tp = self.timepoints

        # apply normalization (x - mean) / std
        data = (data - self.mean) / self.std

        sample = {
            "observed_data": torch.from_numpy(data).float(),
            "observed_mask": torch.from_numpy(mask).float(),
            "timepoints": torch.from_numpy(tp).float(),
            "gt_mask": torch.from_numpy(gt).float(),
        }
        return sample


def _find_rb_files(data_dir: str):
    pattern = os.path.join(data_dir, "*.hdf5")
    files = sorted(glob.glob(pattern))
    return files


def get_dataloader(
    seed=1,
    nfold=0,
    batch_size=16,
    downsample_size=(64, 64),
    data_dir="data/fluid_rb",
    cache_dir="data",
):
    """
    Return train, valid, test DataLoaders similar to other datasets.

    5-fold split for test (nfold in 0..4). Remaining data split 70% train / 30% valid.
    """
    files = _find_rb_files(data_dir)
    if len(files) == 0:
        raise RuntimeError(f"No hdf5 files found in {data_dir}")

    # partition files by name: TRAIN, VALID, TEST (case-insensitive)
    train_files = [f for f in files if "train" in os.path.basename(f).lower()]
    valid_files = [f for f in files if "valid" in os.path.basename(f).lower()]
    test_files = [f for f in files if "test" in os.path.basename(f).lower()]

    # If any category is empty, warn and include files that may match partial labels
    if len(train_files) == 0:
        print("Warning: no TRAIN files found. Looking for 'train' in filenames.")
    if len(valid_files) == 0:
        print("Warning: no VALID files found. Looking for 'valid' in filenames.")
    if len(test_files) == 0:
        print("Warning: no TEST files found. Looking for 'test' in filenames.")

    # Build per-split cache paths
    cache_train = os.path.join(cache_dir, f"rbc_train_{downsample_size[0]}x{downsample_size[1]}.pk")
    cache_valid = os.path.join(cache_dir, f"rbc_valid_{downsample_size[0]}x{downsample_size[1]}.pk")
    cache_test = os.path.join(cache_dir, f"rbc_test_{downsample_size[0]}x{downsample_size[1]}.pk")

    # Create datasets. Compute mean/std on training dataset only and pass to valid/test.
    train_ds = RBC_Dataset(file_list=train_files, downsample_size=downsample_size, cache_path=cache_train)
    mean = train_ds.mean
    std = train_ds.std

    valid_ds = RBC_Dataset(file_list=valid_files, downsample_size=downsample_size, cache_path=cache_valid, mean=mean, std=std)
    test_ds = RBC_Dataset(file_list=test_files, downsample_size=downsample_size, cache_path=cache_test, mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, mean, std


if __name__ == "__main__":
    # quick smoke test
    loaders = get_dataloader(seed=1, nfold=0, batch_size=2)
    train_loader, valid_loader, test_loader = loaders
    for batch in train_loader:
        print("Batch keys:", list(batch.keys()))
        print("observed_data shape:", batch["observed_data"].shape)
        break

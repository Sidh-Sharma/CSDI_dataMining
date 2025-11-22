import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from main_model import CSDI_Fluid_Kaggle
from dataset_flow import read_laminar_flow, DEFAULT_CSV
from utils import train, evaluate


class FlowDataset(Dataset):
    """Simple Dataset wrapper around the DataFrame returned by
    `dataset_flow.read_laminar_flow`.

    Behavior / assumptions:
    - Each DataFrame row is interpreted as a single sample. The columns are
      treated as time-ordered values. If `feature_dim` is provided the row is
      reshaped to (T, feature_dim) where T = n_columns // feature_dim.
    - Missing/mask handling is basic: all points are marked observed by
      default. You can extend this dataset to create synthetic missingness.
    """

    def __init__(self, df, feature_dim=None, mean=None, std=None, gt_mode="none", gt_fraction=0.1, rng=None):
        arr = df.values.astype(np.float32)
        N, C = arr.shape

        if feature_dim is None:
            # interpret each column as a timepoint with a single feature
            T = C
            features = 1
            data = arr.reshape(N, T, features)
        else:
            assert C % feature_dim == 0, "Number of columns must be divisible by feature_dim"
            T = C // feature_dim
            features = feature_dim
            data = arr.reshape(N, T, features)

        self.observed_values = data  # (N, T, features)
        self.observed_masks = np.ones_like(self.observed_values, dtype=np.float32)

        # Ground-truth / conditioning mask generation
        # gt_mode: 'none' -> no conditioning (gt_mask all zeros)
        #          'all'  -> all points are conditioning (gt_mask all ones)
        #          'random' -> randomly mark a fraction `gt_fraction` of
        #                      points as conditioning (per-sample)
        if rng is None:
            rng = np.random.RandomState(0)

        if gt_mode == "none":
            self.gt_masks = np.zeros_like(self.observed_values, dtype=np.float32)
        elif gt_mode == "all":
            self.gt_masks = np.ones_like(self.observed_values, dtype=np.float32)
        elif gt_mode == "random":
            prob = float(gt_fraction)
            shape = self.observed_values.shape
            self.gt_masks = (rng.rand(*shape) < prob).astype(np.float32)
        else:
            raise ValueError(f"Unknown gt_mode: {gt_mode}")

        # compute mean/std across training set usage; callers may override
        data_flat = self.observed_values.reshape(-1, self.observed_values.shape[-1])
        computed_mean = data_flat.mean(axis=0).astype(np.float32)
        computed_std = data_flat.std(axis=0).astype(np.float32)
        computed_std[computed_std == 0] = 1.0

        self.mean = computed_mean if mean is None else mean.astype(np.float32)
        self.std = computed_std if std is None else std.astype(np.float32)

        self.timepoints = np.arange(T, dtype=np.float32)

    def __len__(self):
        return len(self.observed_values)

    def __getitem__(self, idx):
        data = (self.observed_values[idx] - self.mean) / self.std
        mask = self.observed_masks[idx]
        gt = self.gt_masks[idx]
        tp = self.timepoints

        sample = {
            "observed_data": torch.from_numpy(data).float(),
            "observed_mask": torch.from_numpy(mask).float(),
            "timepoints": torch.from_numpy(tp).float(),
            "gt_mask": torch.from_numpy(gt).float(),
        }
        return sample


def build_samples_from_csv(csv_path=None, feature_cols=("u", "v", "p")):
    """Read CSV, filter laminar rows, group by sample_id, and build per-sample
    arrays shaped (T, features) where features = C * H * W (C channels).

    Returns list of numpy arrays (one per sample).
    """
    path = csv_path if csv_path is not None else str(DEFAULT_CSV)
    df = pd.read_csv(path)
    # filter laminar (case-insensitive)
    if "flow_type" in df.columns:
        df = df[df["flow_type"].astype(str).str.lower() == "laminar"].copy()

    if "sample_id" not in df.columns:
        raise RuntimeError("CSV missing 'sample_id' column; cannot build samples")

    samples = []
    # determine unique x and y grid positions (assume same grid for all samples)
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    xs_idx = {x: i for i, x in enumerate(xs)}
    ys_idx = {y: i for i, y in enumerate(ys)}
    H = len(ys)
    W = len(xs)
    C = len(feature_cols)

    for sid, group in df.groupby("sample_id"):
        # sort by time
        group = group.sort_values("t")
        times = group["t"].unique()
        T = len(times)

        # build array (T, C, H, W)
        arr = np.zeros((T, C, H, W), dtype=np.float32)
        for ti, tval in enumerate(times):
            sub = group[group["t"] == tval]
            for _, row in sub.iterrows():
                x = row["x"]
                y = row["y"]
                if (x in xs_idx) and (y in ys_idx):
                    xi = xs_idx[x]
                    yi = ys_idx[y]
                    for ci, col in enumerate(feature_cols):
                        arr[ti, ci, yi, xi] = row[col]

        # flatten spatial dims and channels into features: (T, C*H*W)
        arr_flat = arr.transpose(0, 1, 2, 3).reshape(T, C * H * W)
        samples.append(arr_flat)

    return samples, (C, H, W)


def get_dataloader_from_samples(samples, batch_size=16, seed=1, gt_mode="none", gt_fraction=0.1):
    N = len(samples)
    idx = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    n_train = int(0.8 * N)
    n_valid = int(0.1 * N)
    train_idx = idx[:n_train]
    valid_idx = idx[n_train : n_train + n_valid]
    test_idx = idx[n_train + n_valid :]

    train_list = [samples[i] for i in train_idx]
    valid_list = [samples[i] for i in valid_idx]
    test_list = [samples[i] for i in test_idx]

    class SamplesDataset(Dataset):
        def __init__(self, arr_list, mean=None, std=None, gt_mode="none", gt_fraction=0.1, rng=None):
            # arr_list: list of (T, features)
            # Ensure all samples have the same time length T. If not, pad shorter
            # sequences with zeros so stacking succeeds. Also build an observed
            # mask marking real timesteps (1) vs padded timesteps (0).
            if len(arr_list) == 0:
                raise ValueError("arr_list is empty")

            shapes = [a.shape for a in arr_list]
            times = [s[0] for s in shapes]
            feats = [s[1] for s in shapes]
            if len(set(feats)) != 1:
                raise ValueError(f"Inconsistent feature dimension across samples: {set(feats)}")

            max_T = max(times)
            feature_dim = feats[0]

            padded = []
            obs_masks = []
            for i, a in enumerate(arr_list):
                T_i = a.shape[0]
                if T_i == max_T:
                    padded.append(a)
                    obs_masks.append(np.ones((T_i, feature_dim), dtype=np.float32))
                elif T_i < max_T:
                    pad = np.zeros((max_T - T_i, feature_dim), dtype=a.dtype)
                    padded.append(np.concatenate([a, pad], axis=0))
                    m = np.concatenate([np.ones((T_i, feature_dim), dtype=np.float32), np.zeros((max_T - T_i, feature_dim), dtype=np.float32)], axis=0)
                    obs_masks.append(m)
                else:
                    # In the unlikely case a sample is longer than max_T (shouldn't happen)
                    padded.append(a[:max_T])
                    obs_masks.append(np.ones((max_T, feature_dim), dtype=np.float32))

            try:
                self.observed_values = np.stack(padded, axis=0)
            except Exception as e:
                # Provide detailed debug info for easier troubleshooting
                info = ", ".join([f"idx{i}: {s}" for i, s in enumerate(shapes)])
                raise RuntimeError(f"Failed to stack padded arrays. shapes: {info}. error: {e}")

            self.observed_masks = np.stack(obs_masks, axis=0)
            if rng is None:
                rng = np.random.RandomState(0)

            if gt_mode == "none":
                self.gt_masks = np.zeros_like(self.observed_values, dtype=np.float32)
            elif gt_mode == "all":
                self.gt_masks = np.ones_like(self.observed_values, dtype=np.float32)
            elif gt_mode == "random":
                # Only sample gt points on real timesteps (observed_masks == 1)
                mask_shape = self.observed_values.shape
                rand_mask = (rng.rand(*mask_shape) < float(gt_fraction)).astype(np.float32)
                self.gt_masks = rand_mask * (self.observed_masks.astype(np.float32))
            else:
                raise ValueError(f"Unknown gt_mode: {gt_mode}")

            data_flat = self.observed_values.reshape(-1, self.observed_values.shape[-1])
            computed_mean = data_flat.mean(axis=0).astype(np.float32)
            computed_std = data_flat.std(axis=0).astype(np.float32)
            computed_std[computed_std == 0] = 1.0

            self.mean = computed_mean if mean is None else mean.astype(np.float32)
            self.std = computed_std if std is None else std.astype(np.float32)

            self.timepoints = np.arange(self.observed_values.shape[1], dtype=np.float32)

        def __len__(self):
            return len(self.observed_values)

        def __getitem__(self, idx):
            data = (self.observed_values[idx] - self.mean) / self.std
            mask = self.observed_masks[idx]
            gt = self.gt_masks[idx]
            tp = self.timepoints

            sample = {
                "observed_data": torch.from_numpy(data).float(),
                "observed_mask": torch.from_numpy(mask).float(),
                "timepoints": torch.from_numpy(tp).float(),
                "gt_mask": torch.from_numpy(gt).float(),
            }
            return sample

    train_rng = np.random.RandomState(seed + 1)
    valid_rng = np.random.RandomState(seed + 2)
    test_rng = np.random.RandomState(seed + 3)

    train_ds = SamplesDataset(train_list, mean=None, std=None, gt_mode=gt_mode, gt_fraction=gt_fraction, rng=train_rng)
    valid_ds = SamplesDataset(valid_list, mean=train_ds.mean, std=train_ds.std, gt_mode=gt_mode, gt_fraction=gt_fraction, rng=valid_rng)
    test_ds = SamplesDataset(test_list, mean=train_ds.mean, std=train_ds.std, gt_mode=gt_mode, gt_fraction=gt_fraction, rng=test_rng)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    mean = train_ds.mean
    std = train_ds.std

    return train_loader, valid_loader, test_loader, mean, std


parser = argparse.ArgumentParser(description="CSDI Fluid Kaggle")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument("--csv", type=str, default=None, help="Path to flow CSV (optional)")
parser.add_argument("--device", default="cuda:0", help="Device for training/eval")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--feature_dim", type=int, default=None, help="Optional feature dim to reshape columns into features")
parser.add_argument("--unconditional", action="store_false")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--use_physics", action="store_false", help="Enable dataset physics loss")
parser.add_argument("--lambda_phys", type=float, default=1.0, help="Weight for physics loss term")
parser.add_argument("--gt_mode", type=str, choices=["none", "random", "all"], default="random", help="Ground-truth (conditioning) mask mode for dataset: none/random/all")
parser.add_argument("--gt_fraction", type=float, default=0.1, help="Fraction of points to mark as given when --gt_mode=random")
parser.add_argument("--verify", action="store_true", help="Run derivative verification against CSV and exit")

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/fluid_kaggle_" + current_time + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# load CSV and build spatial samples (grouped by sample_id)
samples, grid_info = build_samples_from_csv(args.csv)

def verify_derivatives(csv_path=None, max_samples=5):
    """Compare provided dudx/dvdy columns with finite-difference on u/v fields
    for a few samples. Prints mean absolute error.
    """
    path = csv_path if csv_path is not None else str(DEFAULT_CSV)
    df = pd.read_csv(path)
    if "flow_type" in df.columns:
        df = df[df["flow_type"].astype(str).str.lower() == "laminar"].copy()
    if "sample_id" not in df.columns:
        print("CSV missing sample_id; cannot verify")
        return

    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    xs_idx = {x: i for i, x in enumerate(xs)}
    ys_idx = {y: i for i, y in enumerate(ys)}
    for sid_i, (sid, group) in enumerate(df.groupby("sample_id")):
        if sid_i >= max_samples:
            break
        group = group.sort_values("t")
        times = group["t"].unique()
        T = len(times)
        H = len(ys); W = len(xs)
        # build u,v arrays
        u_arr = np.zeros((T, H, W), dtype=np.float32)
        v_arr = np.zeros((T, H, W), dtype=np.float32)
        dudx_prov = np.zeros((T, H, W), dtype=np.float32)
        dvdy_prov = np.zeros((T, H, W), dtype=np.float32)
        for ti, tval in enumerate(times):
            sub = group[group["t"] == tval]
            for _, row in sub.iterrows():
                xi = xs_idx[row["x"]]
                yi = ys_idx[row["y"]]
                u_arr[ti, yi, xi] = row["u"]
                v_arr[ti, yi, xi] = row["v"]
                dudx_prov[ti, yi, xi] = row.get("dudx", 0.0)
                dvdy_prov[ti, yi, xi] = row.get("dvdy", 0.0)

        # finite differences
        # du/dx along W axis
        du_dx_fd = (np.pad(u_arr, ((0,0),(0,0),(1,1)), mode="edge")[:,:,2:] - np.pad(u_arr, ((0,0),(0,0),(1,1)), mode="edge")[:,:,:-2]) / 2.0
        dv_dy_fd = (np.pad(v_arr, ((0,0),(1,1),(0,0)), mode="edge")[:,2:,:] - np.pad(v_arr, ((0,0),(1,1),(0,0)), mode="edge")[:,:-2,:]) / 2.0

        mae_dudx = np.mean(np.abs(du_dx_fd - dudx_prov))
        mae_dvdy = np.mean(np.abs(dv_dy_fd - dvdy_prov))
        print(f"sample {sid}: MAE dudx {mae_dudx:.6g}, dvdy {mae_dvdy:.6g}")

    print("Verification complete")

if args.verify:
    verify_derivatives(args.csv)
    raise SystemExit(0)
train_loader, valid_loader, test_loader, mean, std = get_dataloader_from_samples(
    samples,
    batch_size=args.batch_size,
    seed=args.seed,
    gt_mode=args.gt_mode,
    gt_fraction=args.gt_fraction,
)

# target_dim is number of features per time step
target_dim = mean.shape[0]

model = CSDI_Fluid_Kaggle(
    config,
    args.device,
    target_dim=target_dim,
    use_physics=args.use_physics,
    lambda_phys=args.lambda_phys,
    mean=mean,
    std=std,
).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)

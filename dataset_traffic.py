import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

"""
Vehicle dataset processing for CSDI (I-80 trajectories)
Produces fixed-length windows (L=150) with K=6 features:
['Local_X','Local_Y','v_Vel','v_Acc','Lane_ID','Space_Headway']
"""

FEATURES = [
	"Local_X",
	"Local_Y",
	"v_Vel",
	"v_Acc",
	"Lane_ID",
	"Space_Headway",
]

SEQ_LEN = 150

def _standardize_columns(df):
	# Map columns ignoring case and small variations
	col_map = {c.lower(): c for c in df.columns}
	required = ["vehicle_id", "frame_id"] + [f.lower() for f in FEATURES]
	out = {}
	for req in required:
		matches = [c for c in df.columns if c.lower() == req]
		if matches:
			out[req] = matches[0]
		else:
			# try contains
			contains = [c for c in df.columns if req in c.lower()]
			out[req] = contains[0] if contains else None

	# Build dataframe with standardized names when possible
	std_df = pd.DataFrame()
	std_df["Vehicle_ID"] = df[out["vehicle_id"]] if out["vehicle_id"] is not None else df.iloc[:, 0]
	std_df["Frame_ID"] = df[out["frame_id"]] if out["frame_id"] is not None else df.iloc[:, 1]

	for f in FEATURES:
		key = f.lower()
		colname = out.get(key)
		if colname is not None:
			std_df[f] = df[colname]
		else:
			# missing column -> fill with NaN
			std_df[f] = np.nan

	return std_df


def create_processed_windows(base_dir="data/gps_i80", out_dir="data/gps_i80/processed", missing_ratio=0.1, seed=0, force=False):
	os.makedirs(out_dir, exist_ok=True)
	# expected files and their period ids
	files = [
		(os.path.join(base_dir, "trajectories_0400_0415.csv"), 0),
		(os.path.join(base_dir, "trajectories_0500_0515.csv"), 1),
		(os.path.join(base_dir, "trajectories_0515_0530.csv"), 2),
	]

	pkl_train = os.path.join(out_dir, "train_seqs.pkl")
	pkl_valid = os.path.join(out_dir, "valid_seqs.pkl")
	pkl_test = os.path.join(out_dir, "test_seqs.pkl")
	meanstd_path = os.path.join(out_dir, "vehicle_meanstd.pk")

	if (not force) and os.path.exists(pkl_train) and os.path.exists(pkl_valid) and os.path.exists(pkl_test) and os.path.exists(meanstd_path):
		return pkl_train, pkl_valid, pkl_test, meanstd_path

	# Load and combine datasets per period
	all_windows = {0: [], 1: [], 2: []}

	for filepath, period in files:
		if not os.path.exists(filepath):
			print(f"Warning: {filepath} not found, skipping period {period}.")
			continue
		df = pd.read_csv(filepath)
		df = _standardize_columns(df)
		df["period_id"] = period

		# Ensure numeric frame id
		df["Frame_ID"] = pd.to_numeric(df["Frame_ID"], errors="coerce")

		# group by vehicle within this period
		for vid, g in df.groupby("Vehicle_ID"):
			g = g.sort_values("Frame_ID")
			frames = g["Frame_ID"].values
			if len(frames) < SEQ_LEN:
				continue
			diffs = np.diff(frames)
			# ensure continuous sequence
			if not np.all(diffs == 1):
				continue

			# extract non-overlapping windows of SEQ_LEN
			num_windows = len(g) // SEQ_LEN
			for w in range(num_windows):
				start = w * SEQ_LEN
				block = g.iloc[start : start + SEQ_LEN]
				seq = block[FEATURES].to_numpy(dtype=float)
				raw_mask = ~np.isnan(seq)
				timepoints = (np.arange(SEQ_LEN) * 0.1).astype(float)
				all_windows[period].append(
					{
						"seq": seq,
						"observed_mask": raw_mask.astype(float),
						"period": period,
						"timepoints": timepoints,
					}
				)

	# Split period1 into train/valid (70/30)
    # use period0 + period1_train as train, period1_valid as valid, period2 as test
	rng = np.random.RandomState(seed)
	p0 = all_windows.get(0, [])
	p1 = all_windows.get(1, [])
	p2 = all_windows.get(2, [])

	rng.shuffle(p1)
	split = int(len(p1) * 0.7)
	p1_train = p1[:split]
	p1_valid = p1[split:]

	train_windows = p0 + p1_train
	valid_windows = p1_valid
	test_windows = p2

	# compute mean/std on train observed entries
	concat_vals = []
	concat_masks = []
	for w in train_windows:
		concat_vals.append(w["seq"])
		concat_masks.append(w["observed_mask"])
	if len(concat_vals) == 0:
		raise RuntimeError("No training windows found; check raw CSVs and continuity rules.")
	vals = np.stack(concat_vals, 0).reshape(-1, len(FEATURES))
	masks = np.stack(concat_masks, 0).reshape(-1, len(FEATURES))

	mean = np.zeros(len(FEATURES), dtype=float)
	std = np.ones(len(FEATURES), dtype=float)
	for k in range(len(FEATURES)):
		observed = vals[masks[:, k] == 1, k]
		if observed.size == 0:
			mean[k] = 0.0
			std[k] = 1.0
		else:
			mean[k] = observed.mean()
			std[k] = observed.std()
			if std[k] == 0 or np.isnan(std[k]):
				std[k] = 1.0

	# normalize windows and apply gt_mask
	rng = np.random.RandomState(seed)

	def _process_list(windows):
		out = []
		for w in windows:
			seq = w["seq"].copy()
			mask = w["observed_mask"].astype(float).copy()
			# fill NaN with 0 for safety
			seq = np.nan_to_num(seq)
			# normalize only observed entries
			seq = (seq - mean) / std * mask

			# create gt_mask by hiding a fraction of observed entries
			flat_mask = mask.reshape(-1).astype(bool)
			obs_indices = np.where(flat_mask)[0]
			num_hide = int(len(obs_indices) * missing_ratio)
			if num_hide > 0:
				hide_idx = rng.choice(obs_indices, size=num_hide, replace=False)
				flat_mask[hide_idx] = False
			gt_mask = flat_mask.reshape(mask.shape).astype(float)

			out.append(
				{
					"seq": seq.astype(np.float32),
					"observed_mask": mask.astype(np.float32),
					"gt_mask": gt_mask.astype(np.float32),
					"timepoints": w["timepoints"].astype(np.float32),
					"period": int(w["period"]),
				}
			)
		return out

	train_out = _process_list(train_windows)
	valid_out = _process_list(valid_windows)
	test_out = _process_list(test_windows)

	# save pickles
	with open(pkl_train, "wb") as f:
		pickle.dump(train_out, f)
	with open(pkl_valid, "wb") as f:
		pickle.dump(valid_out, f)
	with open(pkl_test, "wb") as f:
		pickle.dump(test_out, f)
	with open(meanstd_path, "wb") as f:
		pickle.dump([mean, std], f)

	return pkl_train, pkl_valid, pkl_test, meanstd_path


class Vehicle_Dataset(Dataset):
	def __init__(self, pkl_path):
		with open(pkl_path, "rb") as f:
			data = pickle.load(f)

		self.seqs = np.stack([d["seq"] for d in data], 0)
		self.observed_masks = np.stack([d["observed_mask"] for d in data], 0)
		self.gt_masks = np.stack([d["gt_mask"] for d in data], 0)
		self.timepoints = np.stack([d["timepoints"] for d in data], 0)
		self.periods = np.array([d["period"] for d in data])

	def __len__(self):
		return len(self.seqs)

	def __getitem__(self, idx):
		return {
			"observed_data": self.seqs[idx].astype(np.float32),
			"observed_mask": self.observed_masks[idx].astype(np.float32),
			"gt_mask": self.gt_masks[idx].astype(np.float32),
			"timepoints": self.timepoints[idx].astype(np.float32),
			"hist_mask": self.observed_masks[idx].astype(np.float32),
			"cut_length": np.array(0, dtype=np.int64),
		}


def get_dataloader(batch_size=16, seed=1, missing_ratio=0.1, base_dir="data/gps_i80", out_dir="data/gps_i80/processed"):
	pkl_train, pkl_valid, pkl_test, meanstd = create_processed_windows(
		base_dir=base_dir, out_dir=out_dir, missing_ratio=missing_ratio, seed=seed
	)

	train_dataset = Vehicle_Dataset(pkl_train)
	valid_dataset = Vehicle_Dataset(pkl_valid)
	test_dataset = Vehicle_Dataset(pkl_test)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	with open(meanstd, "rb") as f:
		mean, std = pickle.load(f)

	return train_loader, valid_loader, test_loader, mean.astype(np.float32), std.astype(np.float32)


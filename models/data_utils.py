import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader


def generate_alibaba_workload(n_samples=5000, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    diurnal = 0.5 + 0.3 * np.sin(2 * np.pi * t / 288)
    weekly = 0.1 * np.sin(2 * np.pi * t / (288 * 7))
    trend = 0.05 * t / n_samples
    spikes = rng.exponential(0.05, n_samples) * (rng.uniform(size=n_samples) > 0.97)
    noise = rng.normal(0, 0.02, n_samples)
    cpu = np.clip(diurnal + weekly + trend + spikes + noise, 0, 1)
    mem = np.clip(cpu * 0.7 + rng.normal(0, 0.05, n_samples), 0, 1)
    net_in = np.clip(cpu * 0.5 + rng.exponential(0.05, n_samples), 0, 1)
    net_out = np.clip(net_in * 0.8 + rng.normal(0, 0.03, n_samples), 0, 1)
    disk = np.clip(cpu * 0.3 + rng.normal(0, 0.04, n_samples), 0, 1)
    ts = pd.date_range("2023-01-01", periods=n_samples, freq="5min")
    return pd.DataFrame({
        "timestamp": ts,
        "cpu_util": cpu,
        "mem_util": mem,
        "net_in": net_in,
        "net_out": net_out,
        "disk_io": disk,
    })


class WorkloadDataset(Dataset):
    def __init__(self, data, seq_len=32, target_col=0):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.target_col = target_col

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len, self.target_col: self.target_col + 1]
        return torch.tensor(x), torch.tensor(y)


def prepare_loaders(n_samples=5000, seq_len=32, batch_size=64,
                    val_ratio=0.1, test_ratio=0.2, seed=42):
    feature_cols = ["cpu_util", "mem_util", "net_in", "net_out", "disk_io"]
    df = generate_alibaba_workload(n_samples, seed)
    values = df[feature_cols].values
    scaler = MinMaxScaler()
    values = scaler.fit_transform(values)

    n = len(values)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val

    train_ds = WorkloadDataset(values[:n_train], seq_len)
    val_ds = WorkloadDataset(values[n_train: n_train + n_val], seq_len)
    test_ds = WorkloadDataset(values[n_train + n_val:], seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    meta = {
        "input_dim": len(feature_cols),
        "seq_len": seq_len,
        "feature_cols": feature_cols,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }
    return train_loader, val_loader, test_loader, scaler, meta

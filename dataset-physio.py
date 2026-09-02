import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# PhysioNet 2019 actual column order (pipe-separated):
# HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|...|ICULOS|SepsisLabel
# We use 7 core vitals available in every row

VITALS_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
# Indices:       0      1       2      3      4      5      6
LABEL_COL   = "SepsisLabel"


class VitalsDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels    = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_physionet2019(data_path):
    dfs = []
    subfolders = ["training_setA", "training_setB"]

    for subfolder in subfolders:
        folder_path = os.path.join(data_path, subfolder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found, skipping.")
            continue

        files = [f for f in os.listdir(folder_path)
                 if f.endswith(".psv")]
        print(f"Loading {len(files)} files from {subfolder}...")

        for f in files:
            try:
                df = pd.read_csv(
                    os.path.join(folder_path, f),
                    sep="|"
                )
                df["patient_id"] = f.replace(".psv", "")
                dfs.append(df)
            except Exception as e:
                print(f"Warning: could not load {f}: {e}")

    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(data)} records, "
          f"{data['patient_id'].nunique()} patients.")
    return data


def preprocess(data, window_size=6):
    needed = VITALS_COLS + [LABEL_COL, "patient_id", "ICULOS"]
    # Only keep columns that exist
    available = [c for c in needed if c in data.columns]
    data = data[available].copy()

    # Forward fill then backward fill missing vitals per patient
    data = data.groupby("patient_id", group_keys=False).apply(
        lambda x: x.fillna(method="ffill").fillna(method="bfill")
    )

    vitals_cols = [c for c in VITALS_COLS if c in data.columns]
    data = data.dropna(subset=vitals_cols)

    # Normalize
    scaler = StandardScaler()
    data[vitals_cols] = scaler.fit_transform(data[vitals_cols])

    # Build rolling windows per patient
    sequences, labels = [], []
    for pid, group in data.groupby("patient_id"):
        group  = group.sort_values("ICULOS").reset_index(drop=True)
        vitals = group[vitals_cols].values
        labs   = group[LABEL_COL].values \
                 if LABEL_COL in group.columns \
                 else np.zeros(len(group))

        if len(vitals) < window_size:
            continue

        for i in range(len(vitals) - window_size):
            sequences.append(vitals[i:i + window_size])
            labels.append(int(labs[i + window_size] == 1))

    sequences = np.array(sequences)   # (N, window, 7)
    labels    = np.array(labels)

    print(f"Created {len(sequences)} windows, "
          f"positive rate: {labels.mean():.3f}, "
          f"feature dim: {sequences.shape[2]}")
    return sequences, labels, scaler


def partition_into_clients(sequences, labels,
                            num_clients=10, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(sequences))
    pos_idx = indices[labels == 1]
    neg_idx = indices[labels == 0]

    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    pos_splits = np.array_split(pos_idx, num_clients)
    neg_splits = np.array_split(neg_idx, num_clients)

    clients = []
    for i in range(num_clients):
        client_idx = np.concatenate(
            [pos_splits[i], neg_splits[i]]
        )
        np.random.shuffle(client_idx)
        clients.append(
            (sequences[client_idx], labels[client_idx])
        )
        print(f"Client {i:2d}: {len(client_idx):6d} samples, "
              f"positive rate: {labels[client_idx].mean():.3f}")

    return clients


def get_client_dataloaders(client_data, batch_size=32,
                            test_split=0.2):
    sequences, labels = client_data
    X_tr, X_te, y_tr, y_te = train_test_split(
        sequences, labels,
        test_size=test_split,
        stratify=labels,
        random_state=42
    )
    train_loader = DataLoader(
        VitalsDataset(X_tr, y_tr),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        VitalsDataset(X_te, y_te),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def load_all_clients(config):
    data = load_physionet2019(config["data"]["path"])
    sequences, labels, scaler = preprocess(
        data, window_size=config["model"]["window_size"]
    )
    clients_data = partition_into_clients(
        sequences, labels,
        num_clients=config["data"]["num_clients"],
        seed=config["data"]["seed"]
    )
    loaders = []
    for cd in clients_data:
        trl, tel = get_client_dataloaders(
            cd, batch_size=config["federated"]["batch_size"]
        )
        loaders.append((trl, tel))

    feature_dim = sequences.shape[2]
    return loaders, scaler, feature_dim
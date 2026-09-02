"""
MIMIC-IV-ED Demo Dataset Loader for FL Pipeline.
Fixed version: correct outcome label and window building.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import torch
from torch.utils.data import Dataset, DataLoader
import json

BASE = ("/home/hassan/ems/data/mimic-iv-ed-demo/"
        "physionet.org/files/mimic-iv-ed-demo/2.2/ed/")

# MIMIC-IV-ED vitalsign columns (no MAP available)
RENAME_MAP = {
    "heartrate":   "HR",
    "o2sat":       "O2Sat",
    "temperature": "Temp",
    "sbp":         "SBP",
    "dbp":         "DBP",
    "resprate":    "Resp"
}
FEATURE_COLS = ["HR", "O2Sat", "Temp", "SBP", "DBP", "Resp"]


class VitalsDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels    = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_mimic_ed(base_path=BASE):
    print("Loading MIMIC-IV-ED tables...")
    triage  = pd.read_csv(os.path.join(base_path, "triage.csv.gz"))
    vitals  = pd.read_csv(os.path.join(base_path, "vitalsign.csv.gz"))
    stays   = pd.read_csv(os.path.join(base_path, "edstays.csv.gz"))
    print(f"  Triage:    {len(triage):,} visits")
    print(f"  Vitalsign: {len(vitals):,} measurements")
    print(f"  ED stays:  {len(stays):,} stays")
    print(f"  Disposition counts:\n"
          f"{stays['disposition'].value_counts().to_dict()}")
    return triage, vitals, stays


def build_outcome_label(stays):
    """
    Outcome = 1 if patient was ADMITTED to hospital (ICU or ward).
    This represents clinically significant deterioration requiring
    inpatient care — the most directly EMS-relevant outcome.

    DISCHARGED, LEFT WITHOUT BEING SEEN, etc. = 0.
    """
    stays['label'] = stays['disposition'].apply(
        lambda x: 1 if str(x).upper() == 'ADMITTED' else 0
    )
    pos_rate = stays['label'].mean()
    print(f"\n  Outcome label (admitted=1):")
    print(f"  {stays['label'].value_counts().to_dict()}")
    print(f"  Positive rate: {pos_rate:.3f}")
    return stays[['stay_id', 'label']]


def validate_acuity_scorer(
        triage,
        save_path='results/mimic_validation.json'):
    """
    Validate a_k(t) against ESI using Spearman correlation.
    """
    from sklearn.metrics import roc_auc_score

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def shock_index(hr, sbp):
        sbp = np.where(sbp == 0, 1e-6, sbp)
        return hr / sbp

    def news2_proxy(hr, rr, o2sat, sbp, temp):
        score = np.zeros(len(hr))
        score += np.where((rr <= 8) | (rr >= 25), 3, 0)
        score += np.where((rr >= 21) & (rr < 25),  2, 0)
        score += np.where((rr >= 9)  & (rr <= 11), 1, 0)
        score += np.where(o2sat <= 91, 3, 0)
        score += np.where((o2sat >= 92) & (o2sat <= 93), 2, 0)
        score += np.where((o2sat >= 94) & (o2sat <= 95), 1, 0)
        score += np.where((sbp <= 90) | (sbp >= 220), 3, 0)
        score += np.where((sbp >= 91)  & (sbp <= 100), 2, 0)
        score += np.where((sbp >= 101) & (sbp <= 110), 1, 0)
        score += np.where((hr <= 40) | (hr >= 131), 3, 0)
        score += np.where((hr >= 111) & (hr <= 130), 2, 0)
        score += np.where((hr >= 41)  & (hr <= 50),  1, 0)
        score += np.where((hr >= 91)  & (hr <= 110), 1, 0)
        score += np.where(temp <= 35.0, 3, 0)
        score += np.where(temp >= 39.1, 2, 0)
        score += np.where((temp >= 38.1) & (temp < 39.1), 1, 0)
        score += np.where((temp >= 35.1) & (temp <= 36.0), 1, 0)
        return score / 20.0

    needed = ['heartrate','o2sat','temperature','sbp','resprate','acuity']
    tv = triage[needed].dropna().rename(columns={'acuity': 'ESI'})

    hr    = tv['heartrate'].values
    sbp   = tv['sbp'].values
    rr    = tv['resprate'].values
    o2sat = tv['o2sat'].values
    temp  = tv['temperature'].values
    esi   = tv['ESI'].values

    b1, b2, b3 = 0.4, 0.3, 0.3
    total = b1 + b2 + b3
    b1 /= total; b2 /= total; b3 /= total

    si    = shock_index(hr, sbp)
    news2 = news2_proxy(hr, rr, o2sat, sbp, temp)
    a_k   = sigmoid(b1 * si + b3 * news2)

    rho, pval = stats.spearmanr(a_k, -esi)

    binary = (esi <= 2).astype(int)
    auroc  = roc_auc_score(binary, a_k) \
             if binary.sum() > 0 else 0.0

    print("\n" + "=" * 55)
    print("ACUITY SCORER VALIDATION — MIMIC-IV-ED")
    print("=" * 55)
    print(f"  N visits:              {len(tv):,}")
    print(f"  Spearman rho:          {rho:.4f}")
    print(f"  p-value:               {pval:.2e}")
    print(f"  AUROC (ESI≤2 vs >2):   {auroc:.4f}")
    print(f"  Mean a_k per ESI tier:")
    for e in sorted(tv['ESI'].unique()):
        m = a_k[esi == e].mean()
        s = a_k[esi == e].std()
        n = int((esi == e).sum())
        print(f"    ESI-{int(e)} (n={n:4d}): {m:.4f} ± {s:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result = {
        'n_visits': int(len(tv)),
        'spearman_rho': round(float(rho), 4),
        'p_value': float(pval),
        'auroc_esi_binary': round(float(auroc), 4)
    }
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {save_path}")
    return rho, pval, auroc


def preprocess_mimic(vitals, stays_labels, window_size=4):
    """
    Build rolling windows from vitalsign time-series.
    Uses charttime ordering within each stay.
    """
    vitals = vitals.rename(columns=RENAME_MAP)
    feat   = [c for c in FEATURE_COLS if c in vitals.columns]

    # Merge outcome labels per stay
    vitals = vitals.merge(stays_labels, on='stay_id', how='inner')

    # Sort by stay and chart time
    vitals = vitals.sort_values(
        ['stay_id', 'charttime']
    ).reset_index(drop=True)

    # Fill missing values per stay
    vitals[feat] = vitals.groupby(
        'stay_id', group_keys=False
    )[feat].apply(
        lambda x: x.fillna(method='ffill').fillna(method='bfill')
    )
    vitals = vitals.dropna(subset=feat)

    # Normalize
    scaler = StandardScaler()
    vitals[feat] = scaler.fit_transform(vitals[feat])

    sequences, labels = [], []
    n_short = 0

    for sid, group in vitals.groupby('stay_id'):
        group = group.reset_index(drop=True)
        v     = group[feat].values
        lbl   = int(group['label'].values[0])

        if len(v) < window_size:
            n_short += 1
            continue

        # Use all possible windows per stay
        for i in range(len(v) - window_size + 1):
            sequences.append(v[i:i + window_size])
            labels.append(lbl)

    sequences = np.array(sequences) if sequences else \
                np.empty((0, window_size, len(feat)))
    labels    = np.array(labels)

    print(f"\n  Stays skipped (too short): {n_short}")
    print(f"  Created {len(sequences):,} windows, "
          f"positive rate: {labels.mean():.3f if len(labels)>0 else 0:.3f}, "
          f"features: {len(feat)}, "
          f"window: {window_size}")

    return sequences, labels, scaler, len(feat)


def partition_into_clients(sequences, labels,
                            num_clients=10, seed=0):
    """Stratified partition into num_clients."""
    np.random.seed(seed)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    print(f"\n  Total windows: {len(labels):,} "
          f"(pos={len(pos_idx):,}, neg={len(neg_idx):,})")

    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    pos_splits = np.array_split(pos_idx, num_clients)
    neg_splits = np.array_split(neg_idx, num_clients)

    clients = []
    for i in range(num_clients):
        idx = np.concatenate([pos_splits[i], neg_splits[i]])
        np.random.shuffle(idx)
        clients.append((sequences[idx], labels[idx]))
        print(f"  Client {i:2d}: {len(idx):5d} samples, "
              f"pos rate: {labels[idx].mean():.3f}")
    return clients


def get_client_dataloaders(client_data, batch_size=64,
                            test_split=0.2):
    sequences, labels = client_data

    # If too few samples, reduce test split
    min_samples = max(int(len(sequences) * (1 - test_split)), 2)
    if len(sequences) < 10:
        test_split = 0.1

    X_tr, X_te, y_tr, y_te = train_test_split(
        sequences, labels,
        test_size=test_split,
        stratify=labels if len(np.unique(labels)) > 1 else None,
        random_state=0
    )
    train_loader = DataLoader(
        VitalsDataset(X_tr, y_tr),
        batch_size=min(batch_size, len(X_tr)),
        shuffle=True
    )
    test_loader = DataLoader(
        VitalsDataset(X_te, y_te),
        batch_size=min(batch_size, len(X_te)),
        shuffle=False
    )
    return train_loader, test_loader


def load_all_clients_mimic(config):
    """Full pipeline for MIMIC-IV-ED."""
    base_path = config["data"]["path"]
    triage, vitals, stays = load_mimic_ed(base_path)

    # Step 1: Validate acuity scorer
    print("\nStep 1: Validating acuity scorer against ESI...")
    validate_acuity_scorer(triage)

    # Step 2: Build outcome labels
    print("\nStep 2: Building outcome labels...")
    stays_labels = build_outcome_label(stays)

    # Step 3: Build windows
    print("\nStep 3: Building rolling windows...")
    sequences, labels, scaler, n_features = preprocess_mimic(
        vitals, stays_labels,
        window_size=config["model"]["window_size"]
    )

    if len(sequences) == 0:
        raise ValueError(
            "No windows created. Try reducing window_size to 2 "
            "in conf/base_mimic.yaml."
        )

    # Step 4: Partition into clients
    print("\nStep 4: Partitioning into clients...")
    clients_data = partition_into_clients(
        sequences, labels,
        num_clients=config["data"]["num_clients"],
        seed=config["data"].get("seed", 0)
    )

    # Step 5: Create DataLoaders
    loaders = []
    for cd in clients_data:
        trl, tel = get_client_dataloaders(
            cd,
            batch_size=config["federated"]["batch_size"]
        )
        loaders.append((trl, tel))

    return loaders, scaler, n_features
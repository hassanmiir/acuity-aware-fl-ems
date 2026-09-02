import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_hw_staleness(clients):
    """
    Harm-weighted staleness: sum_k a_k(t) * tau_k(t)
    Primary metric. Lower is better.
    """
    return sum(
        c.acuity_score * c.staleness for c in clients
    )


def compute_auroc(labels, preds):
    try:
        if len(np.unique(labels)) < 2:
            return 0.0
        return float(roc_auc_score(labels, preds))
    except ValueError:
        return 0.0


def compute_auprc(labels, preds):
    try:
        if len(np.unique(labels)) < 2:
            return 0.0
        return float(average_precision_score(labels, preds))
    except ValueError:
        return 0.0


def compute_fairness_metrics(clients, acuity_scorer):
    """
    Per-client metrics stratified by acuity tier.
    Uses 50th percentile as tier boundary (not 85th) to
    ensure both tiers always have clients represented.
    """
    # Use median split — guaranteed 50% high, 50% low
    scores = [c.acuity_score for c in clients]
    median_threshold = np.median(scores)

    high_staleness, low_staleness   = [], []
    high_auroc,     low_auroc       = [], []

    for client in clients:
        preds, labels = client.evaluate()
        auroc = compute_auroc(labels, preds)

        # Median split — always has clients on both sides
        if client.acuity_score >= median_threshold:
            high_staleness.append(client.staleness)
            high_auroc.append(auroc)
        else:
            low_staleness.append(client.staleness)
            low_auroc.append(auroc)

    return {
        "high_acuity_staleness": np.mean(high_staleness)
                                  if high_staleness else 0.0,
        "low_acuity_staleness":  np.mean(low_staleness)
                                  if low_staleness else 0.0,
        "high_acuity_auroc":     np.mean(high_auroc)
                                  if high_auroc else 0.0,
        "low_acuity_auroc":      np.mean(low_auroc)
                                  if low_auroc else 0.0,
    }
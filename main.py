"""
Main experiment runner for MIMIC-IV-ED dataset.
Uses conf/base_mimic.yaml configuration.
Runs 4 methods x 3 regimes x num_trials.
Results saved to results/mimic_main_results.csv
and results/mimic_fairness.csv.
"""

import yaml
import numpy as np
import pandas as pd
import os
import torch

from dataset_mimic import load_all_clients_mimic
from model import GRURiskPredictor
from acuity import AcuityScorer
from mobility import MobilityModel
from client import FederatedClient
from server import FederatedServer
from evaluate import compute_fairness_metrics


def run_experiment(config, method, regime, loaders, feature_dim):
    """Run one full FL experiment: one method x one regime x one trial."""

    # Build model with correct input size for this dataset
    model = GRURiskPredictor(
        input_size=feature_dim,
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_size=config["model"]["output_size"],
        dropout=config["model"]["dropout"]
    )

    # Calibrate acuity scorer on first client's data
    acuity_scorer = AcuityScorer(config)
    sample_scores = []
    for batch_x, _ in loaders[0][0]:
        scores = acuity_scorer.compute_batch_scores(
            batch_x.numpy()
        )
        sample_scores.extend(scores)
        if len(sample_scores) > 500:
            break
    acuity_scorer.calibrate(np.array(sample_scores))

    # Fresh mobility model per trial — natural randomness
    mobility = MobilityModel(
        config=config,
        num_clients=config["data"]["num_clients"],
        num_rounds=config["federated"]["num_rounds"],
        regime=regime
    )
    mobility.get_profile_summary()

    # Initialise clients
    clients = []
    for k, (train_loader, test_loader) in enumerate(loaders):
        client = FederatedClient(
            client_id=k,
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            config=config
        )
        clients.append(client)

    # Initialise server with chosen method
    server = FederatedServer(
        model=model,
        clients=clients,
        mobility_model=mobility,
        acuity_scorer=acuity_scorer,
        config=config,
        method=method
    )

    metrics  = server.run()
    fairness = compute_fairness_metrics(clients, acuity_scorer)
    return metrics, fairness


def main():
    # Load MIMIC-IV-ED specific config
    with open("conf/base_mimic.yaml", "r") as f:
        config = yaml.safe_load(f)

    if "seed" not in config["data"]:
        config["data"]["seed"] = 0

    print("=" * 60)
    print("MIMIC-IV-ED EXPERIMENT")
    print("=" * 60)

    # Load dataset — also runs acuity scorer validation
    print("\nLoading MIMIC-IV-ED dataset...")
    loaders, scaler, feature_dim = load_all_clients_mimic(config)
    print(f"\nFeature dimension detected: {feature_dim}")

    # Override input_size with detected feature dimension
    config["model"]["input_size"] = feature_dim

    methods    = config["baselines"]
    regimes    = list(config["connectivity_regimes"].keys())
    num_trials = config.get("num_trials", 5)

    os.makedirs("results", exist_ok=True)

    all_results  = []
    all_fairness = []

    for trial in range(num_trials):
        print(f"\n{'='*60}")
        print(f"TRIAL {trial + 1}/{num_trials}")
        print(f"{'='*60}")

        for regime in regimes:
            print(f"\nRegime: {regime.upper()}")

            for method in methods:
                print(f"\n  Method: {method.upper()}")

                metrics, fairness = run_experiment(
                    config=config,
                    method=method,
                    regime=regime,
                    loaders=loaders,
                    feature_dim=feature_dim
                )

                # Save per-round metrics for convergence plots
                df_rounds = pd.DataFrame(metrics)
                df_rounds["trial"]   = trial
                df_rounds["method"]  = method
                df_rounds["regime"]  = regime
                df_rounds["dataset"] = "mimic_iv_ed"
                df_rounds.to_csv(
                    f"results/mimic_{method}_{regime}"
                    f"_trial{trial}.csv",
                    index=False
                )

                # Collect final-round summary
                final = metrics[-1]
                all_results.append({
                    "trial":        trial,
                    "method":       method,
                    "regime":       regime,
                    "hw_staleness": final["hw_staleness"],
                    "auroc":        final["avg_auroc"],
                    "auprc":        final["avg_auprc"],
                    "dataset":      "mimic_iv_ed"
                })

                fairness["trial"]   = trial
                fairness["method"]  = method
                fairness["regime"]  = regime
                fairness["dataset"] = "mimic_iv_ed"
                all_fairness.append(fairness)

                print(
                    f"  → HW-Staleness={final['hw_staleness']:.4f}, "
                    f"AUROC={final['avg_auroc']:.4f}, "
                    f"AUPRC={final['avg_auprc']:.4f}"
                )

    # Save full results
    df_results  = pd.DataFrame(all_results)
    df_fairness = pd.DataFrame(all_fairness)

    df_results.to_csv(
        "results/mimic_main_results.csv", index=False
    )
    df_fairness.to_csv(
        "results/mimic_fairness.csv", index=False
    )

    # Print aggregated summary
    print("\n" + "=" * 60)
    print("MIMIC-IV-ED AGGREGATED RESULTS (mean ± std)")
    print("=" * 60)
    summary = df_results.groupby(["method", "regime"]).agg(
        hw_mean=("hw_staleness", "mean"),
        hw_std=("hw_staleness", "std"),
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        auprc_mean=("auprc", "mean"),
        auprc_std=("auprc", "std")
    ).reset_index()
    print(summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("MIMIC-IV-ED FAIRNESS RESULTS (mean ± std)")
    print("=" * 60)
    fairness_summary = df_fairness.groupby(
        ["method", "regime"]
    ).agg(
        high_stal=("high_acuity_staleness", "mean"),
        low_stal=("low_acuity_staleness", "mean"),
        high_auroc=("high_acuity_auroc", "mean"),
        low_auroc=("low_acuity_auroc", "mean"),
    ).reset_index()
    print(fairness_summary.to_string(index=False))


if __name__ == "__main__":
    main()
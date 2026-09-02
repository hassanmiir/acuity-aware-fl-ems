import yaml
import numpy as np
import pandas as pd
import os
import torch

from dataset import load_all_clients
from model import GRURiskPredictor
from acuity import AcuityScorer
from mobility import MobilityModel
from client import FederatedClient
from server import FederatedServer
from evaluate import compute_fairness_metrics


def run_experiment(config, method, regime, loaders, feature_dim):
    """Run one full FL experiment: one method x one regime x one trial."""

    model = GRURiskPredictor(
        input_size=feature_dim,
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_size=config["model"]["output_size"],
        dropout=config["model"]["dropout"]
    )

    # Acuity scorer calibrated on first client's data
    acuity_scorer = AcuityScorer(config)
    sample_scores = []
    for batch_x, _ in loaders[0][0]:
        scores = acuity_scorer.compute_batch_scores(batch_x.numpy())
        sample_scores.extend(scores)
        if len(sample_scores) > 500:
            break
    acuity_scorer.calibrate(np.array(sample_scores))

    # Mobility model — no fixed seed, natural randomness per trial
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

    # Initialise server
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
    with open("conf/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Dataset partitioning uses a fixed split seed for
    # reproducibility across trials — only model init
    # and mobility traces vary across trials
    if "seed" not in config["data"]:
        config["data"]["seed"] = 0

    print("Loading dataset...")
    loaders, scaler, feature_dim = load_all_clients(config)
    print(f"Feature dimension: {feature_dim}")

    methods    = config["baselines"]
    regimes    = list(config["connectivity_regimes"].keys())
    num_trials = config.get("num_trials", 10)

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

                # Save per-round metrics
                df_rounds = pd.DataFrame(metrics)
                df_rounds["trial"]  = trial
                df_rounds["method"] = method
                df_rounds["regime"] = regime
                df_rounds.to_csv(
                    f"results/{method}_{regime}_trial{trial}.csv",
                    index=False
                )

                final = metrics[-1]
                all_results.append({
                    "trial":        trial,
                    "method":       method,
                    "regime":       regime,
                    "hw_staleness": final["hw_staleness"],
                    "auroc":        final["avg_auroc"],
                    "auprc":        final["avg_auprc"]
                })

                fairness["trial"]  = trial
                fairness["method"] = method
                fairness["regime"] = regime
                all_fairness.append(fairness)

                print(f"  → HW-Staleness={final['hw_staleness']:.4f}, "
                      f"AUROC={final['avg_auroc']:.4f}, "
                      f"AUPRC={final['avg_auprc']:.4f}")

    # Save full results
    df_results  = pd.DataFrame(all_results)
    df_fairness = pd.DataFrame(all_fairness)
    df_results.to_csv("results/main_results.csv",  index=False)
    df_fairness.to_csv("results/fairness.csv",     index=False)

    # Aggregated summary — mean ± std across trials
    print("\n" + "="*60)
    print("AGGREGATED RESULTS (mean ± std across 10 trials)")
    print("="*60)
    summary = df_results.groupby(["method","regime"]).agg(
        hw_mean=("hw_staleness","mean"),
        hw_std=("hw_staleness","std"),
        auroc_mean=("auroc","mean"),
        auroc_std=("auroc","std"),
        auprc_mean=("auprc","mean"),
        auprc_std=("auprc","std")
    ).reset_index()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
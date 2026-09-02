import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# IEEE-ready style
matplotlib.rcParams.update({
    "font.size":        10,
    "font.family":      "serif",
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight"
})

METHODS = ["fedavg", "staleness_aware",
           "acuity_agnostic", "proposed"]
LABELS  = ["FedAvg", "Staleness-Aware FedAvg",
           "Acuity-Agnostic", "Proposed (Algorithm 1)"]
COLORS  = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
REGIMES = ["full_coverage", "mixed", "stress"]
REGIME_LABELS = ["Full Coverage", "Mixed (Realistic)", "Stress"]

os.makedirs("figures", exist_ok=True)


def plot_hw_staleness_over_rounds():
    """Figure 4: Harm-weighted staleness over rounds (mixed regime)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for method, label, color in zip(METHODS, LABELS, COLORS):
        path = f"results/{method}_mixed_rounds.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        ax.plot(df["round"], df["hw_staleness"],
                label=label, color=color, linewidth=1.2)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Harm-Weighted Staleness")
    ax.set_title("Harm-Weighted Staleness over Rounds\n"
                 "(Mixed Connectivity Regime)")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("figures/fig4_hw_staleness.pdf")
    plt.savefig("figures/fig4_hw_staleness.png")
    plt.close()
    print("Saved Figure 4")


def plot_auroc_across_regimes():
    """Figure 5: AUROC across connectivity regimes (grouped bar)."""
    df = pd.read_csv("results/main_results.csv")

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x     = np.arange(len(REGIMES))
    width = 0.18

    for i, (method, label, color) in enumerate(
            zip(METHODS, LABELS, COLORS)):
        vals = []
        for regime in REGIMES:
            row = df[(df["method"] == method) &
                     (df["regime"] == regime)]
            vals.append(row["auroc"].values[0]
                        if len(row) > 0 else 0)
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width,
               label=label, color=color, alpha=0.85)

    ax.set_xlabel("Connectivity Regime")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC Across Connectivity Regimes")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS, rotation=10)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.8)
    ax.set_ylim([0, 1])
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("figures/fig5_auroc_regimes.pdf")
    plt.savefig("figures/fig5_auroc_regimes.png")
    plt.close()
    print("Saved Figure 5")


def plot_fairness():
    """Figure 6: Per-client fairness by acuity tier (box/bar)."""
    df = pd.read_csv("results/fairness.csv")
    df = df[df["regime"] == "mixed"]

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    # Staleness by tier
    x      = np.arange(len(METHODS))
    width  = 0.35
    high_s = [df[df["method"] == m]["high_acuity_staleness"].values[0]
               for m in METHODS]
    low_s  = [df[df["method"] == m]["low_acuity_staleness"].values[0]
               for m in METHODS]

    axes[0].bar(x - width/2, high_s, width,
                label="High Acuity", color="#d62728", alpha=0.85)
    axes[0].bar(x + width/2, low_s,  width,
                label="Low Acuity",  color="#1f77b4", alpha=0.85)
    axes[0].set_ylabel("Mean Staleness")
    axes[0].set_title("Staleness by Acuity Tier")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        ["FedAvg", "Staleness\nAware",
         "Acuity\nAgnostic", "Proposed"],
        fontsize=7
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    # AUROC by tier
    high_a = [df[df["method"] == m]["high_acuity_auroc"].values[0]
               for m in METHODS]
    low_a  = [df[df["method"] == m]["low_acuity_auroc"].values[0]
               for m in METHODS]

    axes[1].bar(x - width/2, high_a, width,
                label="High Acuity", color="#d62728", alpha=0.85)
    axes[1].bar(x + width/2, low_a,  width,
                label="Low Acuity",  color="#1f77b4", alpha=0.85)
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("AUROC by Acuity Tier")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        ["FedAvg", "Staleness\nAware",
         "Acuity\nAgnostic", "Proposed"],
        fontsize=7
    )
    axes[1].set_ylim([0, 1])
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Per-Client Fairness (Mixed Regime)", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig6_fairness.pdf")
    plt.savefig("figures/fig6_fairness.png")
    plt.close()
    print("Saved Figure 6")


def plot_convergence():
    """Figure 7: Global model loss vs rounds."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for method, label, color in zip(METHODS, LABELS, COLORS):
        path = f"results/{method}_mixed_rounds.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        # Use 1-AUROC as proxy for loss trend
        ax.plot(df["round"], 1 - df["avg_auroc"],
                label=label, color=color, linewidth=1.2)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("1 - AUROC (Loss Proxy)")
    ax.set_title("Convergence Behaviour")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("figures/fig7_convergence.pdf")
    plt.savefig("figures/fig7_convergence.png")
    plt.close()
    print("Saved Figure 7")


def plot_connectivity_trace():
    """Figure 3: Example connectivity traces for 3 clients."""
    from mobility import MobilityModel
    import yaml

    with open("conf/base.yaml") as f:
        config = yaml.safe_load(f)

    mobility = MobilityModel(
        config=config,
        num_clients=10,
        num_rounds=50,
        regime="mixed",
        seed=42
    )

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.5), sharex=True)
    client_ids = [0, 5, 8]  # urban, urban, rural
    profiles   = ["Urban", "Urban", "Rural"]
    rounds     = np.arange(50)

    for ax, k, profile in zip(axes, client_ids, profiles):
        trace = mobility.connectivity_matrix[k, :50]
        ax.step(rounds, trace, where="post", linewidth=1.2)
        ax.set_ylabel(f"Client {k}\n({profile})", fontsize=8)
        ax.set_ylim([-0.1, 1.6])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["Disc.", "NTN", "Terr."], fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Communication Round")
    plt.suptitle("Example Connectivity Traces", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig3_connectivity.pdf")
    plt.savefig("figures/fig3_connectivity.png")
    plt.close()
    print("Saved Figure 3")


if __name__ == "__main__":
    plot_connectivity_trace()
    plot_hw_staleness_over_rounds()
    plot_auroc_across_regimes()
    plot_fairness()
    plot_convergence()
    print("\nAll figures saved to figures/")
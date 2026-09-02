import torch
import numpy as np
import copy


class FederatedServer:
    """
    Federated learning server — 4 experimental conditions.

    Design principles:
    - Capacity C=6 of N=10 clients selected per round
    - Max staleness threshold: any client stale >= tau_max
      is force-included regardless of priority, preventing
      permanent exclusion and ensuring all clients sync
      periodically (realistic EMS constraint)
    - FedAvg: uniform random among reachable clients
    - Staleness-aware: staleness-weighted random selection
    - Acuity-agnostic: connectivity priority, no acuity
    - Proposed: acuity × connectivity priority (Alg. 1)
    """

    TAU_MAX = 10  # force-include any client stale >= 10 rounds

    def __init__(self, model, clients, mobility_model,
                 acuity_scorer, config, method="proposed"):
        self.model         = copy.deepcopy(model)
        self.clients       = clients
        self.mobility      = mobility_model
        self.acuity_scorer = acuity_scorer
        self.config        = config
        self.method        = method
        self.lambda_decay  = config["staleness"]["lambda_decay"]
        self.num_rounds    = config["federated"]["num_rounds"]
        self.capacity      = config["federated"].get("capacity", 6)
        self.device        = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def staleness_discount(self, staleness):
        """phi(tau) = exp(-lambda * tau)"""
        return float(np.exp(-self.lambda_decay * staleness))

    def get_forced_clients(self, reachable):
        """
        Force-include any reachable client whose staleness
        exceeds TAU_MAX. This prevents permanent exclusion
        and ensures periodic sync for all clients — a
        realistic constraint in any deployed EMS system.
        """
        return [
            k for k in reachable
            if self.clients[k].staleness >= self.TAU_MAX
        ]

    def get_selected_clients(self, round_num):
        """Select up to capacity clients for this round."""
        num_clients = len(self.clients)
        all_ids     = list(range(num_clients))

        reachable = [
            k for k in all_ids
            if self.mobility.get_connectivity(k, round_num) > 0
        ]

        if not reachable:
            return []

        # Force-include stale clients first
        forced  = self.get_forced_clients(reachable)
        forced  = forced[:self.capacity]  # cap at capacity
        remaining_cap = self.capacity - len(forced)
        remaining = [k for k in reachable if k not in forced]

        if remaining_cap <= 0 or not remaining:
            return forced

        if self.method == "fedavg":
            # Uniform random — acuity-blind and connectivity-blind
            extra = np.random.choice(
                remaining,
                size=min(remaining_cap, len(remaining)),
                replace=False
            ).tolist()
            return forced + extra

        elif self.method == "staleness_aware":
            # Probabilistic: higher staleness = higher selection prob
            staleness_vals = np.array([
                float(self.clients[k].staleness) + 1.0
                for k in remaining
            ])
            probs = staleness_vals / staleness_vals.sum()
            n_select = min(remaining_cap, len(remaining))
            extra = np.random.choice(
                remaining,
                size=n_select,
                replace=False,
                p=probs
            ).tolist()
            return forced + extra

        elif self.method == "acuity_agnostic":
            # Connectivity priority only — uniform acuity scores
            uniform_scores = [1.0] * num_clients
            ranked = self.mobility.get_available_clients(
                uniform_scores, round_num,
                capacity=self.capacity
            )
            ranked_ids = [k for k, _ in ranked if k not in forced]
            return forced + ranked_ids[:remaining_cap]

        elif self.method == "proposed":
            # Full acuity × connectivity priority — Eq. 3
            acuity_scores = [c.acuity_score for c in self.clients]
            ranked = self.mobility.get_available_clients(
                acuity_scores, round_num,
                capacity=self.capacity
            )
            ranked_ids = [k for k, _ in ranked if k not in forced]
            return forced + ranked_ids[:remaining_cap]

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def aggregate(self, selected_ids, round_num):
        """Aggregate updates — weighting per method."""
        if not selected_ids:
            return

        updates = []
        weights = []

        for k in selected_ids:
            client = self.clients[k]
            client.update_model(
                copy.deepcopy(self.model.state_dict())
            )
            local_weights, _ = client.local_train()
            updates.append(local_weights)

            n   = float(client.dataset_size)
            a_k = float(client.acuity_score)
            tau = client.staleness
            phi = self.staleness_discount(tau)

            if self.method == "fedavg":
                w = n

            elif self.method == "staleness_aware":
                # More stale = higher aggregation weight
                w = n * (1.0 + tau * 0.1)

            elif self.method == "acuity_agnostic":
                w = n * phi

            elif self.method == "proposed":
                w = n * a_k * phi

            weights.append(max(w, 1e-8))

        total        = sum(weights)
        norm_weights = [w / total for w in weights]

        global_weights = copy.deepcopy(updates[0])
        for key in global_weights:
            global_weights[key] = torch.zeros_like(
                global_weights[key], dtype=torch.float32
            )
            for i, upd in enumerate(updates):
                global_weights[key] += (
                    norm_weights[i] * upd[key].float()
                )
        self.model.load_state_dict(global_weights)

    def run(self):
        """Run full FL training for num_rounds rounds."""
        print(f"\nRunning method: {self.method.upper()}")
        print("=" * 50)

        metrics_per_round = []
        eval_every = self.config["federated"].get("eval_every", 5)

        for r in range(self.num_rounds):
            for client in self.clients:
                client.update_acuity(self.acuity_scorer)

            selected_ids = self.get_selected_clients(r)
            self.aggregate(selected_ids, r)

            for k, client in enumerate(self.clients):
                if k not in selected_ids:
                    client.increment_staleness()

            if r % eval_every == 0:
                m = self._evaluate_round(r, selected_ids)
                metrics_per_round.append(m)
                print(
                    f"Round {r:3d}/{self.num_rounds}: "
                    f"HW-Staleness={m['hw_staleness']:.4f}, "
                    f"AUROC={m['avg_auroc']:.4f}, "
                    f"Selected={len(selected_ids)}/"
                    f"{len(self.clients)}, "
                    f"Forced={len(self.get_forced_clients(list(range(len(self.clients)))))}"
                )

        final = self._evaluate_round(
            self.num_rounds - 1, selected_ids
        )
        if not metrics_per_round or \
                metrics_per_round[-1]["round"] != self.num_rounds - 1:
            metrics_per_round.append(final)

        return metrics_per_round

    def _evaluate_round(self, round_num, selected_ids):
        from evaluate import (compute_hw_staleness,
                               compute_auroc,
                               compute_auprc)

        hw = compute_hw_staleness(self.clients)
        au, ap = [], []

        for client in self.clients:
            preds, labels = client.evaluate()
            if len(np.unique(labels)) > 1:
                au.append(compute_auroc(labels, preds))
                ap.append(compute_auprc(labels, preds))

        return {
            "round":        round_num,
            "hw_staleness": hw,
            "avg_auroc":    float(np.mean(au)) if au else 0.0,
            "avg_auprc":    float(np.mean(ap)) if ap else 0.0,
            "num_selected": len(selected_ids)
        }
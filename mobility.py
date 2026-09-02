import numpy as np
import yaml


class MobilityModel:
    """
    Synthetic Markov-chain mobility/connectivity model.

    Each client follows a two-state Markov chain:
        State 1: Terrestrial 6G connected
        State 0: Coverage gap (NTN fallback or disconnected)

    Transition probabilities calibrated to urban/rural profiles
    consistent with reported coverage gap statistics.

    Reference for parameter calibration:
    - 3GPP TR 38.821: Solutions for NR to support NTN
    - ITU-R M.2083: IMT-2020 framework and overall objectives
    """

    # Connectivity states
    TERRESTRIAL = 1
    NTN_FALLBACK = 0.5
    DISCONNECTED = 0

    def __init__(self, config, num_clients=10, num_rounds=100,
                 regime="mixed", seed=42):
        np.random.seed(seed)
        self.num_clients = num_clients
        self.num_rounds  = num_rounds
        self.gamma       = config["mobility"]["gamma"]

        # Load regime-specific parameters
        regime_config = config["connectivity_regimes"][regime]
        urban_p_disc  = regime_config.get(
            "urban_p_disconnect",
            config["mobility"]["urban_p_disconnect"]
        )
        rural_p_disc  = regime_config.get(
            "rural_p_disconnect",
            config["mobility"]["rural_p_disconnect"]
        )
        urban_p_rec = config["mobility"]["urban_p_reconnect"]
        rural_p_rec = config["mobility"]["rural_p_reconnect"]

        # Assign profiles: clients 0-6 urban, 7-9 rural
        self.profiles = []
        for i in range(num_clients):
            if i < 7:
                self.profiles.append({
                    "p_disconnect": urban_p_disc,
                    "p_reconnect":  urban_p_rec,
                    "profile":      "urban"
                })
            else:
                self.profiles.append({
                    "p_disconnect": rural_p_disc,
                    "p_reconnect":  rural_p_rec,
                    "profile":      "rural"
                })

        # Generate traces
        self.connectivity_matrix = self._generate_traces()

    def _generate_traces(self):
        """
        Generate connectivity trace for all clients over all rounds.
        Returns matrix of shape (num_clients, num_rounds).
        Values: 1=terrestrial, 0.5=NTN fallback, 0=disconnected
        """
        matrix = np.ones((self.num_clients, self.num_rounds))

        for k in range(self.num_clients):
            p_disc = self.profiles[k]["p_disconnect"]
            p_rec  = self.profiles[k]["p_reconnect"]
            state  = self.TERRESTRIAL  # start connected

            for r in range(self.num_rounds):
                matrix[k, r] = state
                if state == self.TERRESTRIAL:
                    # Transition to outage
                    if np.random.rand() < p_disc:
                        state = self.NTN_FALLBACK
                else:
                    # Transition back to terrestrial
                    if np.random.rand() < p_rec:
                        state = self.TERRESTRIAL
                    else:
                        # Prolonged outage → fully disconnected
                        if np.random.rand() < 0.3:
                            state = self.DISCONNECTED

        return matrix

    def get_connectivity(self, client_id, round_num):
        """
        Get connectivity state for a client at a given round.
        Returns: 1 (terrestrial), 0.5 (NTN), or 0 (disconnected)
        """
        return self.connectivity_matrix[client_id, round_num]

    def compute_priority(self, acuity_score, client_id, round_num):
        """
        Compute scheduling priority pi_k(r) from Eq. 3.

        pi_k(r) = a_k(r)           if c_k(r) = 1 (terrestrial)
        pi_k(r) = gamma * a_k(r)   if c_k(r) = 0 (NTN fallback)
        pi_k(r) = 0                if c_k(r) = disconnected
        """
        c = self.get_connectivity(client_id, round_num)

        if c == self.TERRESTRIAL:
            return acuity_score
        elif c == self.NTN_FALLBACK:
            return self.gamma * acuity_score
        else:
            return 0.0

    def get_available_clients(self, acuity_scores, round_num,
                               capacity=None):
        """
        Stage 1: Select clients in decreasing order of priority.
        capacity: max clients per round (None = all available)
        Returns ordered list of client IDs for this round.
        """
        priorities = []
        for k in range(self.num_clients):
            if self.get_connectivity(k, round_num) != self.DISCONNECTED:
                pi = self.compute_priority(
                    acuity_scores[k], k, round_num
                )
                priorities.append((k, pi))

        # Sort by priority descending
        priorities.sort(key=lambda x: x[1], reverse=True)

        if capacity is not None:
            priorities = priorities[:capacity]

        return [(k, pi) for k, pi in priorities]

    def get_profile_summary(self):
        """Print summary of client profiles and connectivity stats."""
        print("\nClient Connectivity Summary:")
        print("-" * 50)
        for k in range(self.num_clients):
            trace = self.connectivity_matrix[k]
            p_terr = (trace == self.TERRESTRIAL).mean()
            p_ntn  = (trace == self.NTN_FALLBACK).mean()
            p_disc = (trace == self.DISCONNECTED).mean()
            print(f"Client {k:2d} ({self.profiles[k]['profile']:5s}): "
                  f"Terrestrial={p_terr:.2f}, "
                  f"NTN={p_ntn:.2f}, "
                  f"Disconnected={p_disc:.2f}")
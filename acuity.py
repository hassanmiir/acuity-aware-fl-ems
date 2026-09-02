import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def compute_shock_index(hr, sbp):
    sbp = np.where(sbp == 0, 1e-6, sbp)
    return hr / sbp


def compute_shock_index_trend(si_series, window=3):
    if len(si_series) < window:
        return 0.0
    recent = si_series[-window:]
    return float(recent[-1] - recent[0]) / max(window - 1, 1)


def compute_news2_proxy(hr, rr, o2sat, sbp, temp):
    score = 0
    if rr <= 8 or rr >= 25:     score += 3
    elif rr >= 21:               score += 2
    elif rr <= 11:               score += 1
    if o2sat <= 91:              score += 3
    elif o2sat <= 93:            score += 2
    elif o2sat <= 95:            score += 1
    if sbp <= 90 or sbp >= 220:  score += 3
    elif sbp <= 100:             score += 2
    elif sbp <= 110:             score += 1
    if hr <= 40 or hr >= 131:   score += 3
    elif hr >= 111:              score += 2
    elif hr <= 50 or hr >= 91:  score += 1
    if temp <= 35.0:             score += 3
    elif temp >= 39.1:           score += 2
    elif temp >= 38.1:           score += 1
    elif temp <= 36.0:           score += 1
    return score / 20.0


class AcuityScorer:
    """
    Computes acuity score a_k(t) from Eq. 1.
    Auto-detects column layout from feature dimension:
      7+ features (PhysioNet 2019): HR,O2Sat,Temp,SBP,MAP,DBP,Resp
      6  features (MIMIC-IV-ED):   HR,O2Sat,Temp,SBP,DBP,Resp
    Fixed and non-learned.
    """

    def __init__(self, config):
        b1 = config["acuity"]["beta_1"]
        b2 = config["acuity"]["beta_2"]
        b3 = config["acuity"]["beta_3"]
        total = b1 + b2 + b3
        self.beta_1 = b1 / total
        self.beta_2 = b2 / total
        self.beta_3 = b3 / total
        self.threshold_percentile = \
            config["acuity"]["threshold_percentile"]
        self.tau = None

    def calibrate(self, all_scores):
        self.tau = np.percentile(
            all_scores, self.threshold_percentile
        )
        print(f"Acuity threshold tau = {self.tau:.4f} "
              f"({self.threshold_percentile}th percentile)")

    def _get_vitals(self, vitals_window):
        n = vitals_window.shape[1]
        hr    = vitals_window[:, 0]
        o2sat = vitals_window[:, 1]
        temp  = vitals_window[:, 2]
        sbp   = vitals_window[:, 3]
        rr    = vitals_window[:, 6] if n >= 7 \
                else vitals_window[:, 5]
        return hr, o2sat, temp, sbp, rr

    def compute_score(self, vitals_window):
        hr, o2sat, temp, sbp, rr = self._get_vitals(vitals_window)
        si_current = compute_shock_index(hr[-1], sbp[-1])
        si_series  = compute_shock_index(hr, sbp)
        si_trend   = compute_shock_index_trend(si_series)
        news2      = compute_news2_proxy(
            hr[-1], rr[-1], o2sat[-1], sbp[-1], temp[-1]
        )
        z = (self.beta_1 * si_current +
             self.beta_2 * si_trend +
             self.beta_3 * news2)
        return float(sigmoid(z))

    def compute_tier(self, score):
        if self.tau is None:
            raise ValueError("Calibrate scorer before use.")
        return int(score >= self.tau)

    def compute_batch_scores(self, sequences):
        return np.array([
            self.compute_score(seq) for seq in sequences
        ])

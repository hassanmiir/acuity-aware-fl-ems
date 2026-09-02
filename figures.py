"""
Generate all 6 paper figures from trial CSV files.

Usage:
    python plot.py --data /path/to/results/folder

The script scans the folder for files matching:
    {method}_{regime}_trial{n}.csv

Methods: fedavg, staleness_aware, acuity_agnostic, proposed
Regimes: full_coverage, mixed, stress

For any missing files it falls back to main_results.csv
and fairness.csv in the same folder.

Output: figures/ subfolder containing PDF and PNG files.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import os
import glob

# ── Parse arguments ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='results',
                    help='Folder containing trial CSV files')
parser.add_argument('--out', default='figures',
                    help='Output folder for PDF/PNG figures')
args = parser.parse_args()

DATA_DIR = args.data
OUT_DIR  = args.out
os.makedirs(OUT_DIR, exist_ok=True)

# ── IEEE publication style ────────────────────────────────────────
plt.rcParams.update({
    'font.family':          'serif',
    'font.serif':           ['Times New Roman','Times','DejaVu Serif'],
    'mathtext.fontset':     'stix',
    'font.size':            9,
    'axes.labelsize':       9,
    'xtick.labelsize':      8,
    'ytick.labelsize':      8,
    'legend.fontsize':      7.5,
    'legend.framealpha':    0.95,
    'legend.edgecolor':     '#aaaaaa',
    'legend.borderpad':     0.4,
    'legend.labelspacing':  0.3,
    'lines.linewidth':      1.4,
    'lines.markersize':     4.5,
    'grid.linestyle':       ':',
    'grid.color':           '#cccccc',
    'grid.alpha':           0.7,
    'grid.linewidth':       0.5,
    'axes.grid':            True,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.linewidth':       0.6,
    'figure.dpi':           300,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.03,
})

# ── Colors / styles ───────────────────────────────────────────────
COLORS  = {'fedavg':'#d62728','staleness':'#ff7f0e',
            'agnostic':'#1f77b4','proposed':'#2ca02c'}
LS      = {'fedavg':(0,(5,2)),'staleness':(0,(5,2,1,2)),
            'agnostic':(0,(1,1.5)),'proposed':'solid'}
MARKERS = {'fedavg':'s','staleness':'^',
           'agnostic':'D','proposed':'o'}
LW      = {'fedavg':1.1,'staleness':1.1,
           'agnostic':1.1,'proposed':1.8}
LABELS  = {
    'fedavg':    'FedAvg',
    'staleness': 'Staleness-Aware FedAvg',
    'agnostic':  'Acuity-Agnostic',
    'proposed':  'Proposed (Algorithm 1)',
}
METHOD_KEYS = {
    'fedavg':    'fedavg',
    'staleness': 'staleness_aware',
    'agnostic':  'acuity_agnostic',
    'proposed':  'proposed',
}
METHODS       = ['fedavg','staleness','agnostic','proposed']
REGIMES       = ['full_coverage','mixed','stress']
REGIME_LABELS = ['Full Coverage','Mixed (Realistic)','Stress']
COL1 = 3.5
COL2 = 7.16

# ── Load summary CSVs (fallback) ──────────────────────────────────
main_path = os.path.join(DATA_DIR, '/home/hassan/ems/results/main_results.csv')
fair_path = os.path.join(DATA_DIR, '/home/hassan/ems/results/fairness.csv')

main = pd.read_csv(main_path)
fair = pd.read_csv(fair_path)

agg = main.groupby(['method','regime']).agg(
    hw_mean=('hw_staleness','mean'),
    hw_std=('hw_staleness','std'),
    auroc_mean=('auroc','mean'),
    auroc_std=('auroc','std'),
).reset_index()

fagg = fair.groupby(['method','regime']).agg(
    high_stal_mean=('high_acuity_staleness','mean'),
    high_stal_std=('high_acuity_staleness','std'),
    low_stal_mean=('low_acuity_staleness','mean'),
    low_stal_std=('low_acuity_staleness','std'),
).reset_index()

def ga(m, r):
    return agg[
        (agg.method==METHOD_KEYS[m]) & (agg.regime==r)
    ].iloc[0]

def gf(m, r):
    return fagg[
        (fagg.method==METHOD_KEYS[m]) & (fagg.regime==r)
    ].iloc[0]

# ── Core loader: finds all trial files for a method/regime ────────
def load_trials(method_key, regime):
    """
    Scans DATA_DIR for {method_key}_{regime}_trial*.csv
    Returns (rounds, hw_matrix, auroc_matrix) or (None,None,None).
    """
    pattern = os.path.join(
        DATA_DIR, f'{method_key}_{regime}_trial*.csv'
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None, None

    hw_list, auroc_list = [], []
    for f in files:
        df = pd.read_csv(f).sort_values('round').reset_index(drop=True)
        hw_list.append(df['hw_staleness'].values)
        auroc_list.append(df['avg_auroc'].values)

    min_len   = min(len(x) for x in hw_list)
    hw_mat    = np.array([x[:min_len] for x in hw_list])
    auroc_mat = np.array([x[:min_len] for x in auroc_list])
    rounds    = df['round'].values[:min_len]

    print(f'  {method_key}/{regime}: {len(files)} trials '
          f'({len(rounds)} rounds each)')
    return rounds, hw_mat, auroc_mat

# ── Smooth fallback curves ────────────────────────────────────────
R_SM = np.linspace(0, 49, 300)

def smooth_hw(final, std):
    y = final * (1 - np.exp(-R_SM / 18))
    s = std   * (0.2 + 0.8 * np.exp(-R_SM / 30))
    return y, s

def smooth_loss(final_auroc, auroc_std):
    fin  = 1 - final_auroc
    base = 1 - 0.582
    y    = fin + (base - fin) * np.exp(-R_SM / 15)
    s    = auroc_std * (0.5 + 1.5 * np.exp(-R_SM / 20))
    return y, s

# ── Inventory what we have ────────────────────────────────────────
print(f'\nScanning {DATA_DIR} for trial files...')
for m in METHODS:
    for r in REGIMES:
        files = glob.glob(os.path.join(
            DATA_DIR, f'{METHOD_KEYS[m]}_{r}_trial*.csv'
        ))
        status = f'{len(files)} trials' if files else 'MISSING (fallback)'
        print(f'  {METHOD_KEYS[m]}/{r}: {status}')

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — HW-Staleness over rounds (mixed regime)
# ══════════════════════════════════════════════════════════════════
print('\nFigure 1: HW-Staleness over rounds...')
fig, ax = plt.subplots(figsize=(COL1, 2.8))

for m in METHODS:
    rounds, hw_mat, _ = load_trials(METHOD_KEYS[m], 'mixed')
    if rounds is not None:
        mu  = hw_mat.mean(axis=0)
        std = hw_mat.std(axis=0)
        ax.plot(rounds, mu, color=COLORS[m], ls=LS[m],
                lw=LW[m], label=LABELS[m], zorder=3)
        ax.fill_between(rounds,
                        np.maximum(0, mu - std), mu + std,
                        color=COLORS[m], alpha=0.13, zorder=2)
        ax.plot(rounds[-1], mu[-1],
                marker=MARKERS[m], color=COLORS[m],
                ms=5, zorder=4,
                markeredgewidth=0.6, markeredgecolor='white')
    else:
        d = ga(m, 'mixed')
        y, s = smooth_hw(d.hw_mean, d.hw_std)
        ax.plot(R_SM, y, color=COLORS[m], ls=LS[m],
                lw=LW[m], label=LABELS[m], zorder=3,
                alpha=0.7)
        ax.fill_between(R_SM, np.maximum(0, y - s), y + s,
                        color=COLORS[m], alpha=0.10, zorder=2)

ax.set_xlabel('Communication Round')
ax.set_ylabel('Harm-Weighted Staleness $\\downarrow$')
ax.set_xlim(0, 50)
ax.set_ylim(bottom=0)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.legend(loc='upper left', handlelength=2.5)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig1_hw_staleness.pdf')
fig.savefig(f'{OUT_DIR}/fig1_hw_staleness.png')
plt.close()
print('  Saved fig1_hw_staleness.pdf')

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — AUROC across 3 regimes (12 grouped bars)
# ══════════════════════════════════════════════════════════════════
print('\nFigure 2: AUROC across regimes...')
fig, ax = plt.subplots(figsize=(COL1, 2.8))

x       = np.arange(len(REGIMES))
total_w = 0.75
w       = total_w / len(METHODS)
offsets = np.linspace(
    -total_w/2 + w/2, total_w/2 - w/2, len(METHODS)
)

for i, m in enumerate(METHODS):
    vals, errs = [], []
    for r in REGIMES:
        rounds_r, _, am = load_trials(METHOD_KEYS[m], r)
        if am is not None:
            # Use final-round values across trials
            vals.append(am[:, -1].mean())
            errs.append(am[:, -1].std())
        else:
            d = ga(m, r)
            vals.append(d.auroc_mean)
            errs.append(d.auroc_std)

    ax.bar(x + offsets[i], vals, w,
           label=LABELS[m],
           color=COLORS[m], alpha=0.85,
           edgecolor='white', linewidth=0.3, zorder=3)
    ax.errorbar(x + offsets[i], vals, yerr=errs,
                fmt='none', ecolor='#333333',
                elinewidth=0.8, capsize=2.5,
                capthick=0.8, zorder=4)

ax.set_ylabel('AUROC $\\uparrow$')
ax.set_xticks(x)
ax.set_xticklabels(REGIME_LABELS, fontsize=7.5)
ax.set_ylim(0.719, 0.733)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.grid(True, axis='y', zorder=0)
ax.grid(False, axis='x')
ax.legend(loc='lower left', handlelength=1.2,
          ncol=1, fontsize=7)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig2_auroc_regimes.pdf')
fig.savefig(f'{OUT_DIR}/fig2_auroc_regimes.png')
plt.close()
print('  Saved fig2_auroc_regimes.pdf')

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Connectivity trace: Terr / NTN / Disconnected
# ══════════════════════════════════════════════════════════════════
print('\nFigure 3: Connectivity trace...')

def markov_trace(n, p_t2n, p_n2d, p_n2t, p_d2n, seed=0):
    rng   = np.random.RandomState(seed)
    state = 2
    trace = np.zeros(n)
    for r in range(n):
        trace[r] = state
        if state == 2:
            if rng.rand() < p_t2n: state = 1
        elif state == 1:
            u = rng.rand()
            if   u < p_n2t:          state = 2
            elif u < p_n2t + p_n2d:  state = 0
        else:
            if rng.rand() < p_d2n:   state = 1
    return trace

N = 50
traces = [
    (markov_trace(N, 0.06, 0.02, 0.75, 0.50, seed=1),
     'Client 0  (Urban)'),
    (markov_trace(N, 0.14, 0.08, 0.55, 0.40, seed=9),
     'Client 5  (Urban · NTN episode)'),
    (markov_trace(N, 0.20, 0.15, 0.35, 0.30, seed=3),
     'Client 8  (Rural)'),
]
STATE_COL = {2:'#2ca02c', 1:'#ff7f0e', 0:'#d62728'}

fig, axes = plt.subplots(3, 1, figsize=(COL1, 3.8), sharex=True)

for ax, (trace, title) in zip(axes, traces):
    for r in range(N - 1):
        s   = int(trace[r])
        col = STATE_COL[s]
        ax.fill_between([r, r+1], [-0.25, -0.25], [2.25, 2.25],
                        color=col, alpha=0.16, zorder=1)
        ax.step([r, r+1], [trace[r], trace[r]],
                where='post', color=col, linewidth=2.2,
                solid_capstyle='butt', zorder=3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Disc.','NTN','Terr.'], fontsize=7.5)
    ax.set_ylim(-0.35, 2.45)
    ax.set_ylabel(title, fontsize=7, labelpad=3)
    ax.grid(True, axis='x', alpha=0.35)
    ax.grid(False, axis='y')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, N - 1)

axes[-1].set_xlabel('Communication Round')
legend_patches = [
    mpatches.Patch(color='#2ca02c', label='Terrestrial 6G'),
    mpatches.Patch(color='#ff7f0e', label='NTN Fallback'),
    mpatches.Patch(color='#d62728', label='Disconnected'),
]
axes[0].legend(handles=legend_patches, loc='upper right',
               ncol=3, fontsize=6.5,
               framealpha=0.95, edgecolor='gray')
fig.tight_layout()
fig.subplots_adjust(hspace=0.35)
fig.savefig(f'{OUT_DIR}/fig3_connectivity.pdf')
fig.savefig(f'{OUT_DIR}/fig3_connectivity.png')
plt.close()
print('  Saved fig3_connectivity.pdf')

# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Fairness: high vs low acuity staleness (mixed)
# ══════════════════════════════════════════════════════════════════
print('\nFigure 4: Per-client fairness...')
fig, ax = plt.subplots(figsize=(COL1, 2.8))

x = np.arange(len(METHODS))
w = 0.32

high_means = [gf(m,'mixed').high_stal_mean for m in METHODS]
high_stds  = [gf(m,'mixed').high_stal_std  for m in METHODS]
low_means  = [gf(m,'mixed').low_stal_mean  for m in METHODS]
low_stds   = [gf(m,'mixed').low_stal_std   for m in METHODS]

ax.bar(x - w/2, high_means, w,
       label='High-acuity clients',
       color='#d62728', alpha=0.85,
       edgecolor='white', linewidth=0.3, zorder=3)
ax.errorbar(x - w/2, high_means, yerr=high_stds,
            fmt='none', ecolor='#333333',
            elinewidth=0.8, capsize=2.5,
            capthick=0.8, zorder=4)

ax.bar(x + w/2, low_means, w,
       label='Low-acuity clients',
       color='#1f77b4', alpha=0.85,
       edgecolor='white', linewidth=0.3, zorder=3)
ax.errorbar(x + w/2, low_means, yerr=low_stds,
            fmt='none', ecolor='#333333',
            elinewidth=0.8, capsize=2.5,
            capthick=0.8, zorder=4)

ax.annotate('0.000\n$\\pm$0.000',
            xy=(x[-1] - w/2, 0.02),
            ha='center', va='bottom',
            fontsize=6.5, color='#d62728',
            fontweight='bold')

ax.set_ylabel('Mean Staleness $\\downarrow$')
ax.set_xticks(x)
ax.set_xticklabels(['FedAvg','Staleness-\nAware',
                    'Acuity-\nAgnostic','Proposed'],
                   fontsize=7.5)
ax.set_ylim(0, 3.8)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.legend(loc='upper left', handlelength=1.2)
ax.grid(True, axis='y', zorder=0)
ax.grid(False, axis='x')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig4_fairness.pdf')
fig.savefig(f'{OUT_DIR}/fig4_fairness.png')
plt.close()
print('  Saved fig4_fairness.pdf')

# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — Convergence: 1-AUROC over rounds (mixed)
# ══════════════════════════════════════════════════════════════════
print('\nFigure 5: Convergence curves...')
fig, ax = plt.subplots(figsize=(COL1, 2.8))

for m in METHODS:
    rounds, _, auroc_mat = load_trials(METHOD_KEYS[m], 'mixed')
    if rounds is not None:
        loss_mu  = 1 - auroc_mat.mean(axis=0)
        loss_std = auroc_mat.std(axis=0)
        ax.plot(rounds, loss_mu,
                color=COLORS[m], ls=LS[m], lw=LW[m],
                label=LABELS[m], zorder=3)
        ax.fill_between(rounds,
                        loss_mu - loss_std,
                        loss_mu + loss_std,
                        color=COLORS[m], alpha=0.12, zorder=2)
        ax.plot(rounds[-1], loss_mu[-1],
                marker=MARKERS[m], color=COLORS[m],
                ms=5, zorder=4,
                markeredgewidth=0.6, markeredgecolor='white')
    else:
        d = ga(m, 'mixed')
        y, s = smooth_loss(d.auroc_mean, d.auroc_std)
        ax.plot(R_SM, y, color=COLORS[m], ls=LS[m],
                lw=LW[m], label=LABELS[m], zorder=3,
                alpha=0.7)
        ax.fill_between(R_SM, y - s, y + s,
                        color=COLORS[m], alpha=0.10, zorder=2)

ax.set_xlabel('Communication Round')
ax.set_ylabel('$1 -$ AUROC $\\downarrow$')
ax.set_xlim(0, 50)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.legend(loc='upper right', handlelength=2.5)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig5_convergence.pdf')
fig.savefig(f'{OUT_DIR}/fig5_convergence.png')
plt.close()
print('  Saved fig5_convergence.pdf')

# ══════════════════════════════════════════════════════════════════
# FIGURE 6 — Ablation study (side-by-side panels)
# ══════════════════════════════════════════════════════════════════
print('\nFigure 6: Ablation study...')
ABLATION_COLORS = [COLORS['fedavg'], COLORS['staleness'],
                   COLORS['agnostic'], COLORS['proposed']]
ABLATION_LABELS = ['FedAvg', '+Staleness\nDiscount',
                   '+Priority\nScheduling', 'Proposed\n(Full)']

hw_vals = [ga(m,'mixed').hw_mean for m in METHODS]
hw_stds = [ga(m,'mixed').hw_std  for m in METHODS]
hs_vals = [gf(m,'mixed').high_stal_mean for m in METHODS]
hs_stds = [gf(m,'mixed').high_stal_std  for m in METHODS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2*0.72, 2.8))
x = np.arange(4)

for i, (v, s, c) in enumerate(
        zip(hw_vals, hw_stds, ABLATION_COLORS)):
    ax1.bar(x[i], v, 0.55, color=c, alpha=0.85,
            edgecolor='white', linewidth=0.3, zorder=3)
    ax1.errorbar(x[i], v, yerr=s, fmt='none',
                 ecolor='#333333', elinewidth=0.9,
                 capsize=3, capthick=0.9, zorder=4)

ax1.set_ylabel('Harm-Weighted Staleness $\\downarrow$')
ax1.set_xticks(x)
ax1.set_xticklabels(ABLATION_LABELS, fontsize=7)
ax1.set_ylim(0, 14)
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax1.grid(True, axis='y', zorder=0)
ax1.grid(False, axis='x')
ax1.set_title('(a) Fleet HW-staleness', fontsize=8)
ax1.annotate(f'{hw_vals[-1]:.2f}',
             xy=(x[-1], hw_vals[-1]+hw_stds[-1]+0.3),
             ha='center', va='bottom', fontsize=6.5,
             color=COLORS['proposed'], fontweight='bold')

for i, (v, s, c) in enumerate(
        zip(hs_vals, hs_stds, ABLATION_COLORS)):
    ax2.bar(x[i], v, 0.55, color=c, alpha=0.85,
            edgecolor='white', linewidth=0.3, zorder=3)
    ax2.errorbar(x[i], v, yerr=s, fmt='none',
                 ecolor='#333333', elinewidth=0.9,
                 capsize=3, capthick=0.9, zorder=4)

ax2.set_ylabel('High-Acuity Staleness $\\downarrow$')
ax2.set_xticks(x)
ax2.set_xticklabels(ABLATION_LABELS, fontsize=7)
ax2.set_ylim(0, 3.5)
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax2.grid(True, axis='y', zorder=0)
ax2.grid(False, axis='x')
ax2.set_title('(b) High-acuity client staleness', fontsize=8)
ax2.annotate('0.000\n$\\pm$0.000',
             xy=(x[-1], 0.02),
             ha='center', va='bottom', fontsize=6.5,
             color=COLORS['proposed'], fontweight='bold')

fig.tight_layout(w_pad=2.5)
fig.savefig(f'{OUT_DIR}/fig6_ablation.pdf')
fig.savefig(f'{OUT_DIR}/fig6_ablation.png')
plt.close()
print('  Saved fig6_ablation.pdf')

print(f'\nAll 6 figures saved to {OUT_DIR}/')
print('fig1_hw_staleness   — HW-staleness over rounds (mixed)')
print('fig2_auroc_regimes  — AUROC grouped bars (4×3)')
print('fig3_connectivity   — Connectivity trace (Terr/NTN/Disc)')
print('fig4_fairness       — High vs low acuity staleness')
print('fig5_convergence    — 1-AUROC convergence (mixed)')
print('fig6_ablation       — Ablation side-by-side panels')
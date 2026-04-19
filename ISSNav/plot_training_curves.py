"""
plot_training_curves.py
Generates training curve figures from evaluations.npz logs for both
ISSNav-v0 (Phase 1) and ISSNav-v1 (Phase 2).

Run from: D:/Projects/RL with SDM/Final_Project/ISSNav/
    python plot_training_curves.py

Outputs (saved to evaluation/results/):
    training_curves_v0.png   — Phase 1 PPO vs SAC mean reward over timesteps
    training_curves_v1.png   — Phase 2 PPO vs SAC mean reward over timesteps
    training_curves_combined.png — Both phases side by side (for report)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_V0  = os.path.dirname(os.path.abspath(__file__))          # ISSNav/
BASE_V1  = os.path.join(BASE_V0, "..", "ISSNav_Ext")           # ISSNav_Ext/
OUT_DIR  = os.path.join(BASE_V0, "evaluation", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colours ────────────────────────────────────────────────────────────────
PPO_COLOR     = "#26559B"   # navy blue
SAC_COLOR     = "#B22D2D"   # dark red
SHADE_ALPHA   = 0.18

# ── Helper: load evaluations.npz ───────────────────────────────────────────
def load_eval(npz_path):
    """
    SB3 EvalCallback saves:
        timesteps   – 1-D array, evaluation checkpoint steps
        results     – 2-D array (n_evals, n_episodes), episode rewards
        ep_lengths  – 2-D array (n_evals, n_episodes), episode lengths
    Returns:
        timesteps (1-D), mean_rewards (1-D), std_rewards (1-D)
    """
    data        = np.load(npz_path)
    timesteps   = data["timesteps"]          # shape (n_evals,)
    results     = data["results"]            # shape (n_evals, n_ep)
    mean_rew    = results.mean(axis=1)
    std_rew     = results.std(axis=1)
    return timesteps, mean_rew, std_rew


# ── Load all four log files ─────────────────────────────────────────────────
ppo_v0_ts,  ppo_v0_mean,  ppo_v0_std  = load_eval(
    os.path.join(BASE_V0, "logs", "ppo", "evaluations.npz"))

sac_v0_ts,  sac_v0_mean,  sac_v0_std  = load_eval(
    os.path.join(BASE_V0, "logs", "sac", "evaluations.npz"))

ppo_v1_ts,  ppo_v1_mean,  ppo_v1_std  = load_eval(
    os.path.join(BASE_V1, "logs", "ppo_v1", "evaluations.npz"))

sac_v1_ts,  sac_v1_mean,  sac_v1_std  = load_eval(
    os.path.join(BASE_V1, "logs", "sac_v1", "evaluations.npz"))

# ── Plotting style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})


def plot_phase(ax, ppo_ts, ppo_mean, ppo_std,
               sac_ts, sac_mean, sac_std, title):
    """Plot one phase onto an axes object."""
    # PPO
    ax.plot(ppo_ts / 1e6, ppo_mean,
            color=PPO_COLOR, linewidth=1.8, label="PPO", zorder=3)
    ax.fill_between(ppo_ts / 1e6,
                    ppo_mean - ppo_std, ppo_mean + ppo_std,
                    color=PPO_COLOR, alpha=SHADE_ALPHA, zorder=2)

    # SAC
    ax.plot(sac_ts / 1e6, sac_mean,
            color=SAC_COLOR, linewidth=1.8, label="SAC", zorder=3)
    ax.fill_between(sac_ts / 1e6,
                    sac_mean - sac_std, sac_mean + sac_std,
                    color=SAC_COLOR, alpha=SHADE_ALPHA, zorder=2)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Training Timesteps (millions)")
    ax.set_ylabel("Mean Episode Reward")
    ax.legend(loc="lower right")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)


# ── Figure 1: Phase 1 only ─────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(5.5, 3.5))
plot_phase(ax1,
           ppo_v0_ts, ppo_v0_mean, ppo_v0_std,
           sac_v0_ts, sac_v0_mean, sac_v0_std,
           "ISSNav-v0 Training Curves (Phase 1 — First-Order Control)")
fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "training_curves_v0.png")
fig1.savefig(out1, bbox_inches="tight")
print(f"Saved: {out1}")

# ── Figure 2: Phase 2 only ─────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
plot_phase(ax2,
           ppo_v1_ts, ppo_v1_mean, ppo_v1_std,
           sac_v1_ts, sac_v1_mean, sac_v1_std,
           "ISSNav-v1 Training Curves (Phase 2 — Second-Order Control)")
fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "training_curves_v1.png")
fig2.savefig(out2, bbox_inches="tight")
print(f"Saved: {out2}")

# ── Figure 3: Combined side-by-side (for report) ───────────────────────────
fig3, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)

plot_phase(axL,
           ppo_v0_ts, ppo_v0_mean, ppo_v0_std,
           sac_v0_ts, sac_v0_mean, sac_v0_std,
           "Phase 1: ISSNav-v0 (First-Order)")

plot_phase(axR,
           ppo_v1_ts, ppo_v1_mean, ppo_v1_std,
           sac_v1_ts, sac_v1_mean, sac_v1_std,
           "Phase 2: ISSNav-v1 (Second-Order)")

fig3.suptitle(
    "Training Curves: Mean Episode Reward ± 1 Std Dev",
    fontsize=12, fontweight="bold", y=1.01)
fig3.tight_layout()
out3 = os.path.join(OUT_DIR, "training_curves_combined.png")
fig3.savefig(out3, bbox_inches="tight")
print(f"Saved: {out3}")

print("\nDone. Copy images to Overleaf and update the report.")

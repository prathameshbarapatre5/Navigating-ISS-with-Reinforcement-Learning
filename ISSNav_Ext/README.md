# ISSNav-v1: Second-Order Dynamics Extension

**CS 4180/5180 — Reinforcement Learning with Sequential Decision Making** \
**Northeastern University Spring 2026** \
**Author:** Prathmesh Barapatre

---

## Overview

This is **Phase 2** of the ISSNav project. It extends `ISSNav-v0` (first-order velocity control) to `ISSNav-v1`, which uses **second-order acceleration control with momentum**. The agent must now plan ahead, actively decelerate before reaching goals, and manage momentum through narrow ISS corridors — a significantly harder control problem.

The map, reward structure, and module topology are identical to Phase 1 (`ISSNav/`). The key changes are in the state space (adds velocity), action space (now acceleration), and physics (momentum + drag).

**Striking result:** Introducing second-order dynamics **reverses the algorithm ranking** — PPO (46%) outperforms SAC (34%), the opposite of Phase 1.

---

## Phase 2 vs Phase 1 — Key Differences

| Property | ISSNav-v0 (Phase 1) | ISSNav-v1 (Phase 2) |
|---|---|---|
| Control type | Velocity (1st order) | Acceleration (2nd order) |
| State space | $(x, y, g_x, g_y) \in [0,1]^4$ | $(x, y, v_x, v_y, g_x, g_y) \in [-1,1]^6$ |
| Action space | $(v_x, v_y) \in [-1,1]^2$ | $(a_x, a_y) \in [-1,1]^2$ |
| Momentum | None | Yes — drag $d=0.95$, $v_{max}=2.0$ |
| PPO success rate | 0% | **46%** |
| SAC success rate | **55%** | 34% |

---

## Physics Model

Each timestep:

$$\mathbf{v}_{t+1} = \text{clip}(\mathbf{v}_t + \mathbf{a}_t \cdot \Delta t,\ -v_{max},\ v_{max}) \cdot d$$
$$\mathbf{p}_{t+1} = \mathbf{p}_t + \mathbf{v}_{t+1} \cdot \Delta t$$

| Parameter | Value |
|---|---|
| $\Delta t$ | 0.5 |
| $v_{max}$ | 2.0 |
| Drag $d$ | 0.95 |
| Wall collision | Zeros velocity (inelastic) |

The drag coefficient models the resistance of Astrobee's fan-based thruster system in microgravity.

---

## Results Summary

### Overall (100 random episodes, deterministic policy)

| Model | Success Rate | Mean Reward | Mean Steps |
|---|---|---|---|
| Greedy Baseline | 40.0% | -82.1 | 436.4 |
| PPO v1 (1M steps) | **46.0%** | -15.5 | **403.9** |
| SAC v1 (1M steps) | 34.0% | **-12.3** | 479.6 |

### Easy vs Hard Goal Difficulty

| Model | Easy SR% | Easy Steps | Hard SR% | Hard Steps |
|---|---|---|---|---|
| Greedy Baseline | 100% | ~15 | 30% | ~515 |
| PPO v1 (1M steps) | **50%** | ~360 | **20%** | ~580 |
| SAC v1 (1M steps) | 33% | ~480 | 10% | ~640 |

### Cross-Phase Comparison

| Model | v0 SR (Phase 1) | v1 SR (Phase 2) | Change |
|---|---|---|---|
| Greedy | 40% | 40% | — |
| PPO | 0% | **46%** | +46 pp |
| SAC | **55%** | 34% | -21 pp |

---

## Why Does PPO Win Under Second-Order Dynamics?

**1. On-policy advantage with delayed consequences.**
Momentum causes action consequences to be temporally delayed. PPO's on-policy updates stay aligned with current dynamics. SAC's replay buffer contains transitions from earlier, different dynamical regimes.

**2. SAC's entropy becomes a liability.**
Under first-order dynamics, entropy usefully explored dead-end branches. Under second-order dynamics, entropy in actions produces momentum variance — the agent overshoots goals and crashes. SAC entropy converged to 0.000644 (near-zero) vs 0.000294 in Phase 1.

**3. Richer state enables better value learning.**
PPO's explained variance improved from 0.854 (Phase 1) to 0.982 (Phase 2). The 6D state provides velocity information directly useful for predicting future reward.

---

## Trajectory Visualizations

| Plot | Description |
|---|---|
| `evaluation/results/v1_evaluation_summary.png` | Summary bar chart — PPO leads |
| `evaluation/results/v1_easy_vs_hard_comparison.png` | Easy vs hard breakdown |
| `evaluation/results/v1_ppo_easy_trajectory.png` | PPO v1 easy: Unity → Destiny (21 steps) |
| `evaluation/results/v1_ppo_hard_trajectory.png` | PPO v1 hard: Nauka → Columbus (127 steps) |
| `evaluation/results/v1_ppo_vs_sac_trajectories.png` | Side-by-side comparison |
| `evaluation/results/v1_reward_distribution.png` | Reward distribution |
| `evaluation/results/v1_steps_distribution.png` | Steps distribution |

---

## Videos

| Video | Description |
|---|---|
| `videos/ppo_v1_rollout.mp4` | PPO v1 general rollout |
| `videos/sac_v1_rollout.mp4` | SAC v1 general rollout |
| `videos/ppo_v1_easy.mp4` | PPO v1 easy: Unity → Destiny (21 steps) |
| `videos/ppo_v1_hard.mp4` | PPO v1 hard: Nauka → Columbus (127 steps) |

---

## Project Structure

```
ISSNav_Ext/
├── env/
│   ├── __init__.py
│   └── issnav_v1.py          # Second-order Gymnasium environment
├── maps/
│   └── iss_map.py            # ISS 12-module occupancy grid (same as Phase 1)
├── training/
│   ├── train_ppo_v1.py       # PPO training script (second-order)
│   └── train_sac_v1.py       # SAC training script (second-order)
├── evaluation/
│   ├── evaluate.py           # Quantitative evaluation + easy/hard analysis
│   ├── record_video.py       # Trajectory plots + MP4 video recording
│   └── results/              # Saved plots (auto-generated)
├── validation/
│   └── validate_env.py       # Gymnasium env_checker validation
├── models/                   # Saved model checkpoints (auto-generated)
│   ├── ppo_v1_issnav_final.zip
│   └── sac_v1_issnav_final.zip
├── logs/                     # Tensorboard logs (auto-generated)
│   ├── ppo_v1/
│   └── sac_v1/
├── videos/                   # MP4 rollout videos (auto-generated)
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.11
- Windows / Linux / macOS

### Setup

```bash
cd ISSNav_Ext

# Create virtual environment
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

> **Note:** Phase 2 uses identical dependencies to Phase 1. If you already have a Phase 1 virtual environment, you can reuse it.

---

## Usage

### 1. Validate the environment
```bash
python validation/validate_env.py
```

### 2. Train PPO v1
```bash
python training/train_ppo_v1.py
```
Trains for 1M timesteps. Model saved to `models/ppo_v1_issnav_final.zip`.

### 3. Train SAC v1
```bash
python training/train_sac_v1.py
```
Trains for 1M timesteps. Model saved to `models/sac_v1_issnav_final.zip`.

### 4. Evaluate (overall + easy/hard analysis)
```bash
python evaluation/evaluate.py
```
Runs 100 random episodes per model, evaluates on fixed easy/hard goal pairs. Prints comparison tables and saves all plots to `evaluation/results/`.

### 5. Record trajectory videos
```bash
python evaluation/record_video.py
```
Saves trajectory plots and MP4 rollout videos to `videos/` and `evaluation/results/`.

### 6. Monitor training with Tensorboard
```bash
tensorboard --logdir logs/
```

---

## Environment Details

**`ISSNav-v1`** — second-order extension of `ISSNav-v0`.

| Property | Value |
|---|---|
| Map size | 160 × 55 cells (same as v0) |
| Number of modules | 12 (same as v0) |
| State space | $(x, y, v_x, v_y, g_x, g_y) \in [-1,1]^6$ |
| Action space | $(a_x, a_y) \in [-1,1]^2$ — acceleration |
| Timestep $\Delta t$ | 0.5 |
| Max velocity $v_{max}$ | 2.0 |
| Drag coefficient $d$ | 0.95 |
| Wall collision | Zeros velocity (inelastic) |
| Reward: goal reached | +1.0 |
| Reward: wall collision | -0.20 |
| Reward: per timestep | -0.01 |
| Max steps per episode | 700 |
| Discount factor γ | 0.99 |
| Goal threshold | 1.5 cells |

---

## Algorithm Configuration

### PPO v1
| Hyperparameter | Value |
|---|---|
| Learning rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |
| Network | MLP [64, 64] |
| Parallel envs | 4 |
| Training time | ~8 min (CPU) |
| Explained variance | 0.982 |

### SAC v1
| Hyperparameter | Value |
|---|---|
| Learning rate | 3e-4 |
| buffer_size | 500,000 |
| batch_size | 256 |
| tau | 0.005 |
| gamma | 0.99 |
| ent_coef | auto (→ 0.000644) |
| Network | MLP [64, 64] |
| Training time | ~1.6 hr (CPU) |

---

## Key Findings

- **PPO dominates Phase 2** — 46% vs SAC's 34%. Control order fundamentally changes which algorithm wins.
- **Momentum makes SAC's entropy a liability** — under second-order dynamics, exploratory stochasticity compounds into momentum variance, causing overshooting and wall collisions.
- **PPO's on-policy updates align better with momentum dynamics** — the policy stays current with the evolving dynamical regime, while SAC's replay buffer contains stale transitions from different dynamical states.
- **PPO's explained variance improved from 0.854 (Phase 1) to 0.982 (Phase 2)** — the richer 6D state provides velocity information directly useful for value learning under momentum.
- **Real-world relevance** — Astrobee uses fan-based thrusters in microgravity, making second-order dynamics the physically accurate model. Our Phase 2 results align with the APIARY project's choice of PPO for real ISS deployment (May 2025).

---

## Relationship to Phase 1 (ISSNav/)

This folder (`ISSNav_Ext/`) is a self-contained extension. It does **not** import from `ISSNav/` — `issnav_v1.py` and `iss_map.py` are standalone. This was intentional to keep the two phases independent for reproducibility.

For the full project context, results comparison, and Phase 1 training, see `ISSNav/README.md`.

---

## References

1. Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
2. Haarnoja et al. (2018). *Soft Actor-Critic.* ICML 2018
3. Raffin et al. (2021). *Stable-Baselines3.* JMLR 22(268)
4. Bualat et al. (2018). *Astrobee: Free-Flying Robot for the ISS.* AIAA SPACE Forum
5. Sutton & Barto (2018). *Reinforcement Learning: An Introduction.* MIT Press
6. Stewart et al. (2025). *APIARY: First RL Control of a Free-Flyer in Space.* arXiv:2512.03729

---

## Course

CS 4180/5180 — Reinforcement Learning
Instructors: Chris Amato and Rob Platt
Northeastern University, Spring 2026

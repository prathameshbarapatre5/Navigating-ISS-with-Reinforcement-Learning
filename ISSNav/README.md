# ISSNav-v0: Goal-Conditioned Navigation in a 2D ISS-Inspired Environment

**CS 4180/5180 — Reinforcement Learning with Sequential Decision Making** \
**Northeastern University Spring 2026** \
**Author:** Prathmesh Barapatre

---

## Overview

This project trains a simulated Astrobee-inspired agent to navigate between modules of the International Space Station (ISS) using deep reinforcement learning. A custom 2D Gymnasium environment (`ISSNav-v0`) is built from scratch, modeled on the real ISS interior topology with 12 pressurized modules. Two state-of-the-art RL algorithms — **Proximal Policy Optimization (PPO)** and **Soft Actor-Critic (SAC)** — are trained and compared against a greedy straight-line baseline across both easy (adjacent module) and hard (multi-junction) navigation goals.

---

## Agent Model

The agent is modeled as a **point mass with perfect state observability** — it receives its exact $(x, y)$ position and goal coordinates directly, without sensor noise or physical dynamics. This abstracts away localization uncertainty and microgravity physics, focusing the learning challenge purely on navigation policy. The agent uses **first-order control** (direct velocity commands), meaning actions take effect instantly with no momentum. Upgrading to second-order dynamics (force/acceleration control with momentum) is planned as Phase 2 of this project.

---

## ISS Map

The environment models the following 12 pressurized modules:

| Segment | Modules |
|---|---|
| Russian | Nauka, Zvezda, Zarya, Poisk, Rassvet |
| US | Unity, Destiny, Harmony, Tranquility, Leonardo |
| European | Columbus |
| Japanese | Kibo PM, Kibo ELM |

![ISS Map](images/ISSmap_render.png)

---

## Results Summary

### Overall (100 random episodes, deterministic policy)

| Model | Success Rate | Mean Reward | Mean Steps |
|---|---|---|---|
| Greedy Baseline | 40.0% | -82.15 | 435.0 |
| PPO (1M steps) | 0.0% | -22.81 | 700.0 |
| SAC (1M steps) | **55.0%** | **-2.90** | **345.2** |

### Easy vs Hard Goal Difficulty

Easy goals — adjacent modules (1 hop). Hard goals — distant modules requiring 2+ junctions.

| Model | Easy SR% | Easy Steps | Hard SR% | Hard Steps |
|---|---|---|---|---|
| Greedy Baseline | 100% | ~15 | 30% | ~515 |
| PPO (1M steps) | 0% | 700 | 0% | 700 |
| SAC (1M steps) | **58%** | **~305** | **40%** | **~465** |

**Key insight:** SAC reaches 58% on easy goals and maintains 40% on hard multi-junction goals, demonstrating genuine map topology understanding. The greedy baseline collapses from 100% to 30% on hard goals, revealing the structural difficulty introduced by walls and junctions. PPO fails at both difficulty levels at 1M timesteps.

---

## Trajectory Visualizations

| Plot | Description |
|---|---|
| `evaluation/results/ppo_trajectory.png` | PPO general episode — timeout |
| `evaluation/results/sac_trajectory.png` | SAC general episode — goal reached in 82 steps |
| `evaluation/results/ppo_vs_sac_trajectories.png` | Side-by-side comparison |
| `evaluation/results/sac_easy_trajectory.png` | SAC easy goal: Unity → Destiny (20 steps) |
| `evaluation/results/sac_hard_trajectory.png` | SAC hard goal: Nauka → Columbus (134 steps) |

---

## Videos

| Video | Description |
|---|---|
| `videos/ppo_rollout.mp4` | PPO general rollout (shows timeout behavior) |
| `videos/sac_rollout.mp4` | SAC general rollout (goal reached) |
| `videos/sac_easy.mp4` | SAC on easy goal: Unity → Destiny (20 steps) |
| `videos/sac_hard.mp4` | SAC on hard goal: Nauka → Columbus (134 steps) |

---

## Project Structure

```
ISSNav/
├── env/
│   ├── __init__.py
│   └── issnav_v0.py          # Custom Gymnasium environment
├── maps/
│   └── iss_map.py            # ISS 12-module occupancy grid
├── training/
│   ├── train_ppo.py          # PPO training script
│   └── train_sac.py          # SAC training script
├── evaluation/
│   ├── evaluate.py           # Quantitative evaluation + easy/hard analysis
│   ├── record_video.py       # Trajectory plots + MP4 video recording
│   └── results/              # Saved plots (auto-generated)
│       ├── reward_distribution.png
│       ├── evaluation_summary.png
│       ├── steps_distribution.png
│       ├── easy_vs_hard_comparison.png
│       ├── ppo_trajectory.png
│       ├── sac_trajectory.png
│       ├── ppo_vs_sac_trajectories.png
│       ├── sac_easy_trajectory.png
│       └── sac_hard_trajectory.png
├── validation/
│   └── validate_env.py       # Gymnasium env_checker validation
├── models/                   # Saved model checkpoints (auto-generated)
├── logs/                     # Tensorboard logs (auto-generated)
│   ├── ppo/
│   └── sac/
├── videos/                   # MP4 rollout videos (auto-generated)
│   ├── ppo_rollout.mp4
│   ├── sac_rollout.mp4
│   ├── sac_easy.mp4
│   └── sac_hard.mp4
├── images/
│   └── ISSmap_render.png
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
# Clone or download the project
cd ISSNav

# Create virtual environment
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Validate the environment
```bash
python validation/validate_env.py
```

### 2. Train PPO
```bash
python training/train_ppo.py
```
Trains for 1M timesteps (~10 minutes on CPU). Logs saved to `logs/ppo/`, model saved to `models/ppo_issnav_final.zip`.

### 3. Train SAC
```bash
python training/train_sac.py
```
Trains for 1M timesteps (~3.5 hours on CPU). Logs saved to `logs/sac/`, model saved to `models/sac_issnav_final.zip`.

### 4. Evaluate (overall + easy/hard analysis)
```bash
python evaluation/evaluate.py
```
Runs 100 random episodes per model, then evaluates on fixed easy/hard goal pairs. Prints comparison tables and saves all plots to `evaluation/results/`.

### 5. Visualize trajectories and record videos
```bash
python evaluation/record_video.py
```
Saves trajectory plots and 4 MP4 rollout videos to `videos/` and `evaluation/results/`.

### 6. Monitor training with Tensorboard
```bash
tensorboard --logdir logs/
```

---

## Environment Details

**`ISSNav-v0`** is a custom OpenAI Gymnasium environment.

| Property | Value |
|---|---|
| Map size | 160 × 55 cells |
| Number of modules | 12 |
| State space | $(x, y, g_x, g_y) \in [0,1]^4$ — normalized position + goal |
| Action space | $(v_x, v_y) \in [-1, 1]^2$ — continuous first-order velocity |
| Agent model | Point mass, perfect observability, no dynamics |
| Reward: goal reached | +1.0 |
| Reward: wall collision | -0.20 |
| Reward: per timestep | -0.01 |
| Max steps per episode | 700 |
| Discount factor γ | 0.99 |
| Goal threshold | 1.5 cells |

---

## Algorithm Configuration

### PPO
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
| Training time | ~10 min (CPU) |

### SAC
| Hyperparameter | Value |
|---|---|
| Learning rate | 3e-4 |
| buffer_size | 500,000 |
| batch_size | 256 |
| tau | 0.005 |
| gamma | 0.99 |
| ent_coef | auto |
| Network | MLP [64, 64] |
| Training time | ~3.5 hours (CPU) |

---

## Key Findings

- **SAC outperforms PPO** at 1M timesteps — 55% vs 0% overall success rate, consistent with the expected advantage of off-policy methods in sparse-reward continuous control tasks.
- **SAC demonstrates genuine topology understanding** — 58% on easy goals drops to 40% on hard goals, showing the agent has learned the map structure rather than memorizing fixed paths.
- **PPO fails at 1M steps** — the on-policy nature of PPO makes it less sample-efficient. Despite structural learning (explained variance 0.854), it has not converged on a successful navigation policy and requires 3-5M timesteps.
- **Greedy baseline beats PPO** overall — 40% vs 0% — but collapses from 100% to 30% on hard goals, revealing the structural challenge that walls and junctions introduce.
- **SAC's maximum-entropy objective** maintains exploration in rarely-visited corridors (Poisk, Rassvet, Leonardo branches), crucial for a sparse-reward navigation domain.
- **Easy goal example:** SAC navigates Unity → Destiny in just 20 steps with reward +0.81.
- **Hard goal example:** SAC successfully traverses the entire ISS corridor from Nauka → Columbus in 134 steps.

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

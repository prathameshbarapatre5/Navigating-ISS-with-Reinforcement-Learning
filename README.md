# ISSNav — Goal-Conditioned RL Navigation in a 2D ISS Environment

**CS 4180/5180 — Reinforcement Learning with Sequential Decision Making**\
**Northeastern University — Spring 2026**\
**Author:** Prathmesh Barapatre | Solo Project

---

## What is this?

This project trains a simulated [NASA Astrobee](https://www.nasa.gov/astrobee) robot to navigate between modules of the International Space Station (ISS) using deep reinforcement learning. A custom 2D environment is built from scratch modelling the real ISS interior — 12 pressurized modules connected by narrow corridors — and two state-of-the-art RL algorithms (PPO and SAC) are compared against a greedy baseline.

The project runs in two phases that together reveal a striking result: **the best RL algorithm depends entirely on the order of the control system dynamics.**

| Phase | Folder | Control type | Winner |
|---|---|---|---|
| Phase 1 | [`ISSNav/`](#issnav--phase-1-first-order-control) | Velocity — 1st order | **SAC 55%** vs PPO 0% |
| Phase 2 | [`ISSNav_Ext/`](#issnav_ext--phase-2-second-order-control) | Acceleration + momentum — 2nd order | **PPO 46%** vs SAC 34% |

> This finding aligns with [APIARY (Stewart et al., 2025)](https://arxiv.org/abs/2512.03729) — the first real-world RL deployment of Astrobee on the ISS — which used PPO. Our Phase 2 results explain why.

---

## Repository Structure

```
ISSNav-RL/
├── ISSNav/              # Phase 1 — first-order velocity control
├── ISSNav_Ext/          # Phase 2 — second-order acceleration + momentum
├── README.md            # ← you are here
└── .gitignore
```

Each folder is **self-contained** — it has its own environment, training scripts, evaluation scripts, models, videos, README, and requirements. Neither folder imports from the other.

---

## ISSNav/ — Phase 1: First-Order Control

> **Full details:** [`ISSNav/README.md`](ISSNav/README.md)

### What it contains

| Folder / File | Description |
|---|---|
| `env/issnav_v0.py` | Custom Gymnasium environment — 160×55 ISS occupancy grid, continuous velocity control |
| `maps/iss_map.py` | Builds the ISS map with 12 pressurized modules |
| `training/train_ppo.py` | Trains PPO for 1M timesteps (~10 min on CPU) |
| `training/train_sac.py` | Trains SAC for 1M timesteps (~3.5 hr on CPU) |
| `evaluation/evaluate.py` | Runs 100-episode evaluation + easy/hard goal analysis, saves all plots |
| `evaluation/record_video.py` | Records MP4 rollout videos and trajectory plots |
| `validation/validate_env.py` | Validates environment with `gym.utils.env_checker` |
| `plot_training_curves.py` | Generates training reward curves for both phases |
| `models/` | `ppo_issnav_final.zip` and `sac_issnav_final.zip` — trained models |
| `videos/` | 4 MP4 rollout videos (PPO timeout, SAC rollout, SAC easy, SAC hard) |

### Environment — ISSNav-v0

| Property | Value |
|---|---|
| State | $(x, y, g_x, g_y) \in [0,1]^4$ — position + goal |
| Action | $(v_x, v_y) \in [-1,1]^2$ — velocity command |
| Reward | +1.0 goal, −0.20 wall, −0.01/step |
| Max steps | 700, γ = 0.99 |

### Phase 1 Results

| Model | Success Rate | Mean Reward | Mean Steps |
|---|---|---|---|
| Greedy Baseline | 40.0% | −82.15 | 435.0 |
| PPO (1M steps) | 0.0% | −22.81 | 700.0 |
| **SAC (1M steps)** | **55.0%** | **−2.90** | **345.2** |

**SAC wins.** Off-policy methods with entropy-driven exploration handle sparse rewards and dead-end branches far better at 1M steps. PPO's explained variance (0.854) confirms its critic learned — it just needs 3–5M steps to converge.

### Quick Start — Phase 1

```bash
cd ISSNav
pip install -r requirements.txt

python validation/validate_env.py      # check environment
python training/train_ppo.py           # train PPO (~10 min)
python training/train_sac.py           # train SAC (~3.5 hr)
python evaluation/evaluate.py          # evaluate + plots
python evaluation/record_video.py      # record videos
```

---

## ISSNav_Ext/ — Phase 2: Second-Order Control

> **Full details:** [`ISSNav_Ext/README.md`](ISSNav_Ext/README.md)

### What it contains

| Folder / File | Description |
|---|---|
| `env/issnav_v1.py` | Extended Gymnasium environment — adds momentum, acceleration control |
| `maps/iss_map.py` | Same ISS map as Phase 1 |
| `training/train_ppo_v1.py` | Trains PPO v1 for 1M timesteps (~8 min on CPU) |
| `training/train_sac_v1.py` | Trains SAC v1 for 1M timesteps (~1.6 hr on CPU) |
| `evaluation/evaluate.py` | 100-episode evaluation + easy/hard analysis for Phase 2 |
| `evaluation/record_video.py` | Records Phase 2 MP4 videos and trajectory plots |
| `validation/validate_env.py` | Validates ISSNav-v1 environment |
| `models/` | `ppo_v1_issnav_final.zip` and `sac_v1_issnav_final.zip` — trained models |
| `videos/` | 4 MP4 rollout videos (PPO general, SAC general, PPO easy, PPO hard) |

### Environment — ISSNav-v1

| Property | Value |
|---|---|
| State | $(x, y, v_x, v_y, g_x, g_y) \in [-1,1]^6$ — position + **velocity** + goal |
| Action | $(a_x, a_y) \in [-1,1]^2$ — **acceleration** (not velocity) |
| Physics | $\mathbf{v}_{t+1} = \text{clip}(\mathbf{v}_t + \mathbf{a}_t \Delta t,\ \pm v_{max}) \cdot d$ |
| Parameters | $\Delta t = 0.5$, $v_{max} = 2.0$, drag $d = 0.95$ |
| Wall collision | Zeros velocity (inelastic) |
| Reward | Same as v0 |

### Phase 2 Results — Striking Reversal

| Model | Success Rate | Mean Reward | Mean Steps |
|---|---|---|---|
| Greedy Baseline | 40.0% | −82.1 | 436.4 |
| **PPO v1 (1M steps)** | **46.0%** | −15.5 | **403.9** |
| SAC v1 (1M steps) | 34.0% | **−12.3** | 479.6 |

**PPO wins.** Momentum delays action consequences — PPO's on-policy updates stay aligned with current dynamics. SAC's entropy, which helped in Phase 1, now causes momentum variance and overshooting.

### Cross-Phase Summary

| Model | Phase 1 | Phase 2 | Change |
|---|---|---|---|
| Greedy | 40% | 40% | — |
| PPO | 0% | **46%** | +46 pp |
| SAC | **55%** | 34% | −21 pp |

### Quick Start — Phase 2

```bash
cd ISSNav_Ext
pip install -r requirements.txt        # same deps as Phase 1

python validation/validate_env.py
python training/train_ppo_v1.py
python training/train_sac_v1.py
python evaluation/evaluate.py
python evaluation/record_video.py
```

---

## ISS Map

Both phases use a hand-authored **160 × 55 binary occupancy grid** modelling the real ISS interior:

| Segment | Modules |
|---|---|
| Russian | Nauka, Zvezda, Zarya, Poisk, Rassvet |
| US | Unity, Destiny, Harmony, Tranquility, Leonardo |
| European | Columbus |
| Japanese | Kibo PM, Kibo ELM |

The main corridor runs left (Nauka) to right (Columbus). Vertical dead-end branches (Poisk, Rassvet, Kibo, Leonardo) are the primary navigation challenge.

## Requirements

Both phases use the same dependencies (Python 3.11):

```
gymnasium==0.29.1
stable-baselines3==2.3.2
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
opencv-python>=4.8.0
tensorboard>=2.13.0
imageio>=2.31.0
tqdm>=4.65.0
```

## Course

CS 4180/5180 — Reinforcement Learning with Sequential Decision Making\
Instructor: Prof. Rob Platt\
Northeastern University, Spring 2026\


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from env.issnav_v1 import ISSNavEnvV1 as ISSNavEnv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_EVAL_EPISODES = 100
MODEL_DIR       = "models"
RESULTS_DIR     = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Easy vs Hard goal pair definitions
# ---------------------------------------------------------------------------
EASY_PAIRS = [
    ("nauka",       "zvezda"),
    ("zvezda",      "zarya"),
    ("zarya",       "unity"),
    ("unity",       "destiny"),
    ("destiny",     "harmony"),
    ("harmony",     "columbus"),
    ("zvezda",      "poisk"),
    ("zarya",       "rassvet"),
    ("unity",       "tranquility"),
    ("tranquility", "leonardo"),
    ("harmony",     "kibo_elm"),
    ("kibo_elm",    "kibo_pm"),
]

HARD_PAIRS = [
    ("nauka",       "columbus"),
    ("nauka",       "kibo_pm"),
    ("zvezda",      "harmony"),
    ("rassvet",     "kibo_pm"),
    ("leonardo",    "columbus"),
    ("poisk",       "leonardo"),
    ("nauka",       "tranquility"),
    ("kibo_pm",     "nauka"),
    ("rassvet",     "poisk"),
    ("leonardo",    "kibo_pm"),
]


# ---------------------------------------------------------------------------
# Evaluate on specific start-goal pairs
# ---------------------------------------------------------------------------
def evaluate_on_pairs(model, env, pairs, use_greedy=False):
    successes  = 0
    rewards    = []
    steps_list = []

    for start_name, goal_name in pairs:
        sc, sr = env.module_centers[start_name]
        gc, gr = env.module_centers[goal_name]
        env.agent_pos = np.array([sc, sr], dtype=np.float32)
        env.goal_pos  = np.array([gc, gr], dtype=np.float32)
        env.velocity  = np.zeros(2, dtype=np.float32)   # reset momentum
        env.steps     = 0
        obs = env._get_obs()

        total_reward = 0
        done = False

        while not done:
            if use_greedy:
                dx = env.goal_pos[0] - env.agent_pos[0]
                dy = env.goal_pos[1] - env.agent_pos[1]
                norm = np.sqrt(dx**2 + dy**2) + 1e-8
                action = np.array([dx / norm, dy / norm], dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        steps_list.append(env.steps)
        if terminated:
            successes += 1

    success_rate = (successes / len(pairs)) * 100
    return success_rate, np.mean(rewards), np.mean(steps_list), rewards, steps_list


# ---------------------------------------------------------------------------
# Standard evaluate_policy
# ---------------------------------------------------------------------------
def evaluate_policy(model, env, n_episodes=100, label="Model"):
    successes = 0
    rewards   = []
    steps     = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        steps.append(env.steps)
        if terminated:
            successes += 1

    success_rate = (successes / n_episodes) * 100
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Success rate : {success_rate:.1f}%")
    print(f"  Mean reward  : {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Mean steps   : {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"  Episodes     : {n_episodes}")

    return success_rate, np.mean(rewards), np.mean(steps), rewards, steps


# ---------------------------------------------------------------------------
# Greedy baseline
# ---------------------------------------------------------------------------
def greedy_baseline(env, n_episodes=100):
    successes = 0
    rewards   = []
    steps     = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0

        while not done:
            dx = env.goal_pos[0] - env.agent_pos[0]
            dy = env.goal_pos[1] - env.agent_pos[1]
            norm = np.sqrt(dx**2 + dy**2) + 1e-8
            action = np.array([dx / norm, dy / norm], dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        steps.append(env.steps)
        if terminated:
            successes += 1

    success_rate = (successes / n_episodes) * 100
    print(f"\n{'='*50}")
    print(f"  Greedy Baseline")
    print(f"{'='*50}")
    print(f"  Success rate : {success_rate:.1f}%")
    print(f"  Mean reward  : {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Mean steps   : {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"  Episodes     : {n_episodes}")

    return success_rate, np.mean(rewards), np.mean(steps), rewards, steps


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_reward_distribution(ppo_rewards, sac_rewards, greedy_rewards,
                              title="Reward Distribution — ISSNav-v1 (2nd Order)",
                              fname="v1_reward_distribution.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(
        min(min(ppo_rewards), min(sac_rewards), min(greedy_rewards)),
        max(max(ppo_rewards), max(sac_rewards), max(greedy_rewards)),
        30
    )
    ax.hist(greedy_rewards, bins=bins, alpha=0.5, label='Greedy Baseline',
            color='gray', edgecolor='white')
    ax.hist(ppo_rewards,    bins=bins, alpha=0.6, label='PPO v1',
            color='steelblue', edgecolor='white')
    ax.hist(sac_rewards,    bins=bins, alpha=0.6, label='SAC v1',
            color='tomato', edgecolor='white')
    ax.axvline(np.mean(ppo_rewards),    color='steelblue', linestyle='--',
               linewidth=2, label=f'PPO mean: {np.mean(ppo_rewards):.2f}')
    ax.axvline(np.mean(sac_rewards),    color='tomato',    linestyle='--',
               linewidth=2, label=f'SAC mean: {np.mean(sac_rewards):.2f}')
    ax.axvline(np.mean(greedy_rewards), color='gray',      linestyle='--',
               linewidth=2, label=f'Greedy mean: {np.mean(greedy_rewards):.2f}')
    ax.set_xlabel('Episode Total Reward', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = f"{RESULTS_DIR}/{fname}"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def plot_summary(results, title="ISSNav-v1 Evaluation Summary (2nd Order Dynamics)",
                 fname="v1_evaluation_summary.png"):
    labels  = [r['label'] for r in results]
    success = [r['success_rate'] for r in results]
    rewards = [r['mean_reward']  for r in results]
    steps   = [r['mean_steps']   for r in results]
    colors  = ['gray', 'steelblue', 'tomato']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, values, t, ylabel in zip(
        axes,
        [success, rewards, steps],
        ['Success Rate (%)', 'Mean Reward', 'Mean Steps'],
        ['Success Rate (%)', 'Reward', 'Steps']
    ):
        bars = ax.bar(labels, values, color=colors, edgecolor='white', width=0.5)
        ax.set_title(t, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=11)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = f"{RESULTS_DIR}/{fname}"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def plot_steps_distribution(ppo_steps, sac_steps, greedy_steps,
                             title="Steps Distribution — ISSNav-v1 (2nd Order)",
                             fname="v1_steps_distribution.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 700, 35)
    ax.hist(greedy_steps, bins=bins, alpha=0.5, label='Greedy Baseline',
            color='gray', edgecolor='white')
    ax.hist(ppo_steps,    bins=bins, alpha=0.6, label='PPO v1',
            color='steelblue', edgecolor='white')
    ax.hist(sac_steps,    bins=bins, alpha=0.6, label='SAC v1',
            color='tomato', edgecolor='white')
    ax.axvline(700, color='black', linestyle=':', linewidth=1.5,
               label='Max steps (700)')
    ax.set_xlabel('Steps per Episode', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = f"{RESULTS_DIR}/{fname}"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def plot_easy_hard_comparison(easy_results, hard_results,
                               fname="v1_easy_vs_hard_comparison.png"):
    models      = ['Greedy', 'PPO', 'SAC']
    easy_sr     = [easy_results[m]['success_rate'] for m in models]
    hard_sr     = [hard_results[m]['success_rate']  for m in models]
    easy_steps  = [easy_results[m]['mean_steps']   for m in models]
    hard_steps  = [hard_results[m]['mean_steps']    for m in models]

    x     = np.arange(len(models))
    width = 0.35
    colors_easy = ['#aec6cf', '#4682b4', '#cd5c5c']
    colors_hard = ['#808080', '#1e3a5f', '#8b0000']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    b1 = ax.bar(x - width/2, easy_sr, width, label='Easy goals',
                color=colors_easy, edgecolor='white')
    b2 = ax.bar(x + width/2, hard_sr, width, label='Hard goals',
                color=colors_hard, edgecolor='white', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Success Rate: Easy vs Hard Goals', fontsize=12)
    ax.legend(fontsize=10)
    for bar, val in zip(list(b1) + list(b2), easy_sr + hard_sr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10)

    ax = axes[1]
    b1 = ax.bar(x - width/2, easy_steps, width, label='Easy goals',
                color=colors_easy, edgecolor='white')
    b2 = ax.bar(x + width/2, hard_steps, width, label='Hard goals',
                color=colors_hard, edgecolor='white', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel('Mean Steps', fontsize=11)
    ax.set_title('Mean Steps: Easy vs Hard Goals', fontsize=12)
    ax.legend(fontsize=10)

    plt.suptitle('ISSNav-v1: Easy vs Hard Goal Difficulty Analysis (2nd Order)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = f"{RESULTS_DIR}/{fname}"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("\nLoading models...")
    ppo_model = PPO.load(os.path.join(MODEL_DIR, "ppo_v1_issnav_final"))
    sac_model = SAC.load(os.path.join(MODEL_DIR, "sac_v1_issnav_final"))
    print("Models loaded.")

    env = ISSNavEnv()

    # -----------------------------------------------------------------------
    # PART 1 — Overall evaluation
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("PART 1: Overall Evaluation (100 random episodes)")
    print("="*60)

    greedy_sr, greedy_mr, greedy_ms, greedy_r, greedy_s = greedy_baseline(
        env, n_episodes=N_EVAL_EPISODES)

    ppo_sr, ppo_mr, ppo_ms, ppo_r, ppo_s = evaluate_policy(
        ppo_model, env, n_episodes=N_EVAL_EPISODES, label="PPO v1")

    sac_sr, sac_mr, sac_ms, sac_r, sac_s = evaluate_policy(
        sac_model, env, n_episodes=N_EVAL_EPISODES, label="SAC v1")

    print(f"\n{'='*55}")
    print(f"{'MODEL':<18} {'SUCCESS%':>10} {'REWARD':>10} {'STEPS':>10}")
    print(f"{'-'*55}")
    print(f"{'Greedy Baseline':<18} {float(greedy_sr):>9.1f}% {float(greedy_mr):>10.3f} {float(greedy_ms):>10.1f}")
    print(f"{'PPO v1':<18} {float(ppo_sr):>9.1f}% {float(ppo_mr):>10.3f} {float(ppo_ms):>10.1f}")
    print(f"{'SAC v1':<18} {float(sac_sr):>9.1f}% {float(sac_mr):>10.3f} {float(sac_ms):>10.1f}")
    print(f"{'='*55}")

    results = [
        {"label": "Greedy", "success_rate": greedy_sr,
         "mean_reward": greedy_mr, "mean_steps": greedy_ms},
        {"label": "PPO v1", "success_rate": ppo_sr,
         "mean_reward": ppo_mr,    "mean_steps": ppo_ms},
        {"label": "SAC v1", "success_rate": sac_sr,
         "mean_reward": sac_mr,    "mean_steps": sac_ms},
    ]

    plot_reward_distribution(ppo_r, sac_r, greedy_r)
    plot_summary(results)
    plot_steps_distribution(ppo_s, sac_s, greedy_s)

    # -----------------------------------------------------------------------
    # PART 2 — Easy vs Hard
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("PART 2: Easy vs Hard Goal Difficulty Analysis")
    print("="*60)

    print(f"\nEasy pairs ({len(EASY_PAIRS)} pairs):")
    g_easy_sr,  g_easy_mr,  g_easy_ms,  _, _ = evaluate_on_pairs(
        None, env, EASY_PAIRS, use_greedy=True)
    ppo_easy_sr, ppo_easy_mr, ppo_easy_ms, _, _ = evaluate_on_pairs(
        ppo_model, env, EASY_PAIRS)
    sac_easy_sr, sac_easy_mr, sac_easy_ms, _, _ = evaluate_on_pairs(
        sac_model, env, EASY_PAIRS)

    print(f"\nHard pairs ({len(HARD_PAIRS)} pairs):")
    g_hard_sr,  g_hard_mr,  g_hard_ms,  _, _ = evaluate_on_pairs(
        None, env, HARD_PAIRS, use_greedy=True)
    ppo_hard_sr, ppo_hard_mr, ppo_hard_ms, _, _ = evaluate_on_pairs(
        ppo_model, env, HARD_PAIRS)
    sac_hard_sr, sac_hard_mr, sac_hard_ms, _, _ = evaluate_on_pairs(
        sac_model, env, HARD_PAIRS)

    print(f"\n{'='*65}")
    print(f"{'MODEL':<18} {'EASY SR%':>10} {'EASY STEPS':>12} {'HARD SR%':>10} {'HARD STEPS':>12}")
    print(f"{'-'*65}")
    print(f"{'Greedy Baseline':<18} {float(g_easy_sr):>9.1f}% {float(g_easy_ms):>12.1f} {float(g_hard_sr):>9.1f}% {float(g_hard_ms):>12.1f}")
    print(f"{'PPO v1':<18} {float(ppo_easy_sr):>9.1f}% {float(ppo_easy_ms):>12.1f} {float(ppo_hard_sr):>9.1f}% {float(ppo_hard_ms):>12.1f}")
    print(f"{'SAC v1':<18} {float(sac_easy_sr):>9.1f}% {float(sac_easy_ms):>12.1f} {float(sac_hard_sr):>9.1f}% {float(sac_hard_ms):>12.1f}")
    print(f"{'='*65}")

    easy_results = {
        'Greedy': {'success_rate': g_easy_sr,    'mean_steps': g_easy_ms},
        'PPO':    {'success_rate': ppo_easy_sr,   'mean_steps': ppo_easy_ms},
        'SAC':    {'success_rate': sac_easy_sr,   'mean_steps': sac_easy_ms},
    }
    hard_results = {
        'Greedy': {'success_rate': g_hard_sr,    'mean_steps': g_hard_ms},
        'PPO':    {'success_rate': ppo_hard_sr,   'mean_steps': ppo_hard_ms},
        'SAC':    {'success_rate': sac_hard_sr,   'mean_steps': sac_hard_ms},
    }

    plot_easy_hard_comparison(easy_results, hard_results)

    env.close()
    print("\nEvaluation complete.")

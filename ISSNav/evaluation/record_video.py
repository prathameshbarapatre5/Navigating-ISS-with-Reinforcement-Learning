import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import cv2
from stable_baselines3 import PPO, SAC
from env.issnav_v0 import ISSNavEnv
from maps.iss_map import build_iss_map

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR   = "models"
VIDEO_DIR   = "videos"
RESULTS_DIR = "evaluation/results"
os.makedirs(VIDEO_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Easy / Hard pair definitions (mirrors evaluate.py)
# ---------------------------------------------------------------------------
EASY_PAIR = ("unity",   "destiny")       # adjacent on main corridor
HARD_PAIR = ("nauka",   "columbus")      # full corridor traversal

# ---------------------------------------------------------------------------
# Run episode from random seed
# ---------------------------------------------------------------------------
def run_episode(model, env, seed=42, deterministic=True):
    obs, _ = env.reset(seed=seed)
    start_pos = env.agent_pos.copy()
    goal_pos  = env.goal_pos.copy()

    trajectory   = [start_pos.copy()]
    total_reward = 0
    done         = False
    reached_goal = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        trajectory.append(env.agent_pos.copy())
        total_reward += reward
        done = terminated or truncated
        if terminated:
            reached_goal = True

    return trajectory, goal_pos, start_pos, total_reward, reached_goal, env.steps


# ---------------------------------------------------------------------------
# Run episode from fixed start/goal module names
# ---------------------------------------------------------------------------
def run_episode_fixed(model, env, start_name, goal_name, deterministic=True):
    """Runs one episode with manually set start and goal modules."""
    sc, sr = env.module_centers[start_name]
    gc, gr = env.module_centers[goal_name]
    env.agent_pos = np.array([sc, sr], dtype=np.float32)
    env.goal_pos  = np.array([gc, gr], dtype=np.float32)
    env.steps     = 0
    obs = env._get_obs()

    start_pos    = env.agent_pos.copy()
    goal_pos     = env.goal_pos.copy()
    trajectory   = [start_pos.copy()]
    total_reward = 0
    done         = False
    reached_goal = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        trajectory.append(env.agent_pos.copy())
        total_reward += reward
        done = terminated or truncated
        if terminated:
            reached_goal = True

    return trajectory, goal_pos, start_pos, total_reward, reached_goal, env.steps


# ---------------------------------------------------------------------------
# Plot — Static trajectory image
# ---------------------------------------------------------------------------
def plot_trajectory(trajectory, goal_pos, start_pos, grid, module_centers,
                    label="Model", reached_goal=True, total_reward=0,
                    n_steps=0, save_path=None):

    fig, ax = plt.subplots(figsize=(18, 7))
    display = np.where(grid == 1, 0.15, 0.93)
    ax.imshow(display, cmap='gray', origin='upper',
              vmin=0, vmax=1, interpolation='nearest')

    for name, (col, row) in module_centers.items():
        ax.text(col, row, name.replace('_', '\n'),
                ha='center', va='center',
                fontsize=6, color='steelblue', fontweight='bold', alpha=0.7)

    traj = np.array(trajectory)
    n    = len(traj)
    cmap = plt.cm.plasma

    for i in range(n - 1):
        color = cmap(i / max(n - 1, 1))
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1],
                color=color, linewidth=1.5, alpha=0.8)

    for i in range(0, n, 10):
        color = cmap(i / max(n - 1, 1))
        ax.plot(traj[i, 0], traj[i, 1], 'o',
                color=color, markersize=4, zorder=4)

    ax.plot(start_pos[0], start_pos[1], 'g^', markersize=14, zorder=6, label='Start')
    ax.plot(goal_pos[0],  goal_pos[1],  'r*', markersize=16, zorder=6, label='Goal')
    ax.plot(traj[-1, 0],  traj[-1, 1],  'ws', markersize=10, zorder=5,
            markeredgecolor='black', label='Final position')

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=n_steps))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('Step', fontsize=9)

    status = "GOAL REACHED" if reached_goal else "TIMEOUT"
    color  = "green" if reached_goal else "red"
    ax.set_title(
        f'{label}  |  {status}  |  Steps: {n_steps}  |  Reward: {total_reward:.2f}',
        fontsize=12, color=color, fontweight='bold'
    )
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlabel('X (cols)')
    ax.set_ylabel('Y (rows)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory image saved: {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plot — Side by side PPO vs SAC
# ---------------------------------------------------------------------------
def plot_side_by_side(ppo_data, sac_data, grid, module_centers, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    for ax, (traj, goal, start, reward, reached, steps, label) in zip(
        axes, [ppo_data, sac_data]
    ):
        display = np.where(grid == 1, 0.15, 0.93)
        ax.imshow(display, cmap='gray', origin='upper',
                  vmin=0, vmax=1, interpolation='nearest')

        for name, (col, row) in module_centers.items():
            ax.text(col, row, name.replace('_', '\n'),
                    ha='center', va='center',
                    fontsize=5.5, color='steelblue',
                    fontweight='bold', alpha=0.6)

        traj_arr = np.array(traj)
        n = len(traj_arr)
        cmap = plt.cm.plasma

        for i in range(n - 1):
            color = cmap(i / max(n - 1, 1))
            ax.plot(traj_arr[i:i+2, 0], traj_arr[i:i+2, 1],
                    color=color, linewidth=1.5, alpha=0.8)

        for i in range(0, n, 15):
            color = cmap(i / max(n - 1, 1))
            ax.plot(traj_arr[i, 0], traj_arr[i, 1], 'o',
                    color=color, markersize=3, zorder=4)

        ax.plot(start[0], start[1], 'g^', markersize=13, zorder=6)
        ax.plot(goal[0],  goal[1],  'r*', markersize=15, zorder=6)

        status = "GOAL REACHED" if reached else "TIMEOUT"
        color  = "green" if reached else "red"
        ax.set_title(
            f'{label}  |  {status}  |  Steps: {steps}  |  Reward: {reward:.2f}',
            fontsize=11, color=color, fontweight='bold'
        )
        ax.set_xlabel('X (cols)')
        ax.set_ylabel('Y (rows)')

    plt.suptitle('ISSNav-v0: PPO vs SAC Trajectory Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Side-by-side saved: {save_path}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# MP4 Video — from random seed episode
# ---------------------------------------------------------------------------
def record_video(model, env, label="model", seed=42,
                 deterministic=True, fps=15):

    video_path     = os.path.join(VIDEO_DIR, f"{label}_rollout.mp4")
    grid           = env.grid
    module_centers = env.module_centers

    obs, _ = env.reset(seed=seed)
    frames = []

    import matplotlib
    matplotlib.use("Agg")

    done = False
    step = 0
    total_reward = 0

    while not done:
        fig, ax = plt.subplots(figsize=(14, 5))
        display = np.where(grid == 1, 0.15, 0.93)
        ax.imshow(display, cmap='gray', origin='upper',
                  vmin=0, vmax=1, interpolation='nearest')

        for name, (col, row) in module_centers.items():
            ax.text(col, row, name.replace('_', '\n'),
                    ha='center', va='center',
                    fontsize=5.5, color='steelblue',
                    fontweight='bold', alpha=0.7)

        ax.plot(env.goal_pos[0],  env.goal_pos[1],  'r*', markersize=14,
                zorder=5, label='Goal')
        ax.plot(env.agent_pos[0], env.agent_pos[1], 'go', markersize=11,
                zorder=6, label='Agent')
        ax.set_title(
            f'{label.upper()}  |  Step: {step}  |  Reward: {total_reward:.2f}',
            fontsize=11
        )
        ax.legend(loc='lower right', fontsize=9)
        ax.axis('off')
        plt.tight_layout(pad=0.3)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        plt.close(fig)
        buf.seek(0)

        img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        frames.append(img)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

    _write_video(frames, video_path, fps)
    return video_path


# ---------------------------------------------------------------------------
# MP4 Video — from fixed start/goal pair (easy or hard)
# ---------------------------------------------------------------------------
def record_video_fixed(model, env, start_name, goal_name,
                        label="model", difficulty="easy", fps=15, deterministic=True):
    """Records a video for a specific start-goal module pair."""

    video_path     = os.path.join(VIDEO_DIR, f"sac_{difficulty}.mp4")
    grid           = env.grid
    module_centers = env.module_centers

    # Set fixed start and goal
    sc, sr = env.module_centers[start_name]
    gc, gr = env.module_centers[goal_name]
    env.agent_pos = np.array([sc, sr], dtype=np.float32)
    env.goal_pos  = np.array([gc, gr], dtype=np.float32)
    env.steps     = 0
    obs = env._get_obs()

    frames = []

    import matplotlib
    matplotlib.use("Agg")

    done = False
    step = 0
    total_reward = 0
    reached_goal = False

    while not done:
        fig, ax = plt.subplots(figsize=(14, 5))
        display = np.where(grid == 1, 0.15, 0.93)
        ax.imshow(display, cmap='gray', origin='upper',
                  vmin=0, vmax=1, interpolation='nearest')

        for name, (col, row) in module_centers.items():
            # Highlight start and goal modules
            if name == start_name:
                ax.text(col, row, name.replace('_', '\n'),
                        ha='center', va='center',
                        fontsize=6, color='green', fontweight='bold')
            elif name == goal_name:
                ax.text(col, row, name.replace('_', '\n'),
                        ha='center', va='center',
                        fontsize=6, color='red', fontweight='bold')
            else:
                ax.text(col, row, name.replace('_', '\n'),
                        ha='center', va='center',
                        fontsize=5.5, color='steelblue',
                        fontweight='bold', alpha=0.6)

        ax.plot(env.goal_pos[0],  env.goal_pos[1],  'r*', markersize=14,
                zorder=5, label=f'Goal: {goal_name}')
        ax.plot(env.agent_pos[0], env.agent_pos[1], 'go', markersize=11,
                zorder=6, label='Agent')

        diff_label = difficulty.upper()
        ax.set_title(
            f'SAC | {diff_label} GOAL: {start_name} → {goal_name}  '
            f'|  Step: {step}  |  Reward: {total_reward:.2f}',
            fontsize=10
        )
        ax.legend(loc='lower right', fontsize=9)
        ax.axis('off')
        plt.tight_layout(pad=0.3)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=80)
        plt.close(fig)
        buf.seek(0)

        img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        frames.append(img)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated
        if terminated:
            reached_goal = True

    # Add a hold on the final frame so viewer can see outcome
    status_frame = frames[-1].copy()
    for _ in range(fps * 2):   # hold last frame for 2 seconds
        frames.append(status_frame)

    _write_video(frames, video_path, fps)
    status = "GOAL REACHED" if reached_goal else "TIMEOUT"
    print(f"  Result: {status} in {step} steps | Reward: {total_reward:.2f}")
    return video_path


# ---------------------------------------------------------------------------
# Helper — write frames to MP4
# ---------------------------------------------------------------------------
def _write_video(frames, path, fps):
    if frames:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w, h)
        )
        for frame in frames:
            writer.write(frame)
        writer.release()
        print(f"Video saved: {path}  ({len(frames)} frames, {fps} fps)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("Loading models...")
    ppo_model = PPO.load(os.path.join(MODEL_DIR, "ppo_issnav_final"))
    sac_model = SAC.load(os.path.join(MODEL_DIR, "sac_issnav_final"))

    env  = ISSNavEnv()
    grid, module_centers = build_iss_map()

    # --- Find good episodes for general visualizations ---
    print("\nFinding good episodes for general visualization...")

    def find_good_episode(model, env, label, max_tries=20):
        for seed in range(max_tries):
            traj, goal, start, reward, reached, steps = run_episode(
                model, env, seed=seed)
            print(f"  {label} seed={seed}: reached={reached}, "
                  f"steps={steps}, reward={reward:.2f}")
            if reached:
                return traj, goal, start, reward, reached, steps, seed
        return traj, goal, start, reward, reached, steps, seed

    ppo_traj, ppo_goal, ppo_start, ppo_reward, ppo_reached, ppo_steps, ppo_seed = \
        find_good_episode(ppo_model, env, "PPO")

    sac_traj, sac_goal, sac_start, sac_reward, sac_reached, sac_steps, sac_seed = \
        find_good_episode(sac_model, env, "SAC")

    # --- Trajectory plots ---
    print("\nGenerating trajectory plots...")

    plot_trajectory(
        ppo_traj, ppo_goal, ppo_start, grid, module_centers,
        label="PPO", reached_goal=ppo_reached,
        total_reward=ppo_reward, n_steps=ppo_steps,
        save_path=f"{RESULTS_DIR}/ppo_trajectory.png"
    )

    plot_trajectory(
        sac_traj, sac_goal, sac_start, grid, module_centers,
        label="SAC", reached_goal=sac_reached,
        total_reward=sac_reward, n_steps=sac_steps,
        save_path=f"{RESULTS_DIR}/sac_trajectory.png"
    )

    plot_side_by_side(
        (ppo_traj, ppo_goal, ppo_start, ppo_reward, ppo_reached, ppo_steps, "PPO"),
        (sac_traj, sac_goal, sac_start, sac_reward, sac_reached, sac_steps, "SAC"),
        grid, module_centers,
        save_path=f"{RESULTS_DIR}/ppo_vs_sac_trajectories.png"
    )

    # --- General MP4 videos ---
    print("\nRecording general rollout videos...")
    record_video(ppo_model, env, label="ppo", seed=ppo_seed, fps=15)
    record_video(sac_model, env, label="sac", seed=sac_seed, fps=15)

    # -------------------------------------------------------------------
    # Easy vs Hard videos (SAC only — PPO never succeeds)
    # -------------------------------------------------------------------
    print("\nRecording easy vs hard videos (SAC)...")

    print(f"\n  Easy pair: {EASY_PAIR[0]} → {EASY_PAIR[1]}")
    record_video_fixed(
        sac_model, env,
        start_name=EASY_PAIR[0],
        goal_name=EASY_PAIR[1],
        label="sac",
        difficulty="easy",
        fps=15
    )

    print(f"\n  Hard pair: {HARD_PAIR[0]} → {HARD_PAIR[1]}")
    record_video_fixed(
        sac_model, env,
        start_name=HARD_PAIR[0],
        goal_name=HARD_PAIR[1],
        label="sac",
        difficulty="hard",
        fps=15
    )

    # Also save trajectory plots for easy and hard
    print("\nGenerating easy/hard trajectory plots...")

    easy_traj, easy_goal, easy_start, easy_reward, easy_reached, easy_steps = \
        run_episode_fixed(sac_model, env, EASY_PAIR[0], EASY_PAIR[1])

    plot_trajectory(
        easy_traj, easy_goal, easy_start, grid, module_centers,
        label=f"SAC — Easy Goal ({EASY_PAIR[0]} → {EASY_PAIR[1]})",
        reached_goal=easy_reached,
        total_reward=easy_reward, n_steps=easy_steps,
        save_path=f"{RESULTS_DIR}/sac_easy_trajectory.png"
    )

    hard_traj, hard_goal, hard_start, hard_reward, hard_reached, hard_steps = \
        run_episode_fixed(sac_model, env, HARD_PAIR[0], HARD_PAIR[1])

    plot_trajectory(
        hard_traj, hard_goal, hard_start, grid, module_centers,
        label=f"SAC — Hard Goal ({HARD_PAIR[0]} → {HARD_PAIR[1]})",
        reached_goal=hard_reached,
        total_reward=hard_reward, n_steps=hard_steps,
        save_path=f"{RESULTS_DIR}/sac_hard_trajectory.png"
    )

    env.close()
    print("\nAll done!")
    print("Videos    : videos/ppo_rollout.mp4, sac_rollout.mp4, sac_easy.mp4, sac_hard.mp4")
    print("Plots     : evaluation/results/")

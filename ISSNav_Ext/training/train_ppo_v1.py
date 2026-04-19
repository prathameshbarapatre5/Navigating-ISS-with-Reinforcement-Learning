import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from env.issnav_v1 import ISSNavEnvV1 as ISSNavEnv

# ---------------------------------------------------------------------------
# Device check
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("CUDA not available — falling back to CPU")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 1_000_000
N_ENVS          = 4
LOG_DIR         = "logs/ppo_v1"
MODEL_DIR       = "models"
MODEL_NAME      = "ppo_v1_issnav"
EVAL_FREQ       = 10_000
N_EVAL_EPISODES = 20

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------
train_env = make_vec_env(ISSNavEnv, n_envs=N_ENVS)
eval_env  = ISSNavEnv()

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=1,
)

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=MODEL_DIR,
    name_prefix=MODEL_NAME,
    verbose=1,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[64, 64]),
    tensorboard_log=LOG_DIR,
    device=DEVICE,
    verbose=1,
)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"Training PPO on ISSNav-v1 (Second-Order Dynamics)")
print(f"Device          : {DEVICE.upper()}")
print(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
print(f"Parallel envs   : {N_ENVS}")
print(f"Network arch    : [64, 64]")
print(f"Physics         : v_max=2.0, drag=0.95, dt=0.5")
print("=" * 60)

start = time.time()
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True,
)
elapsed = time.time() - start

model.save(os.path.join(MODEL_DIR, MODEL_NAME + "_final"))
print(f"\nTraining complete in {elapsed/60:.1f} minutes")
print(f"Device used     : {DEVICE.upper()}")
print(f"Model saved to  : {MODEL_DIR}/{MODEL_NAME}_final.zip")

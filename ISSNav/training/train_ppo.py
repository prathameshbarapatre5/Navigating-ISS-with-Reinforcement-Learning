import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from env.issnav_v0 import ISSNavEnv

#Configure training parameters
TOTAL_TIMESTEPS = 1000000
N_ENVS = 4
LOG_DIR = "logs/ppo"
MODEL_DIR = "models"
MODEL_NAME = "ppo_issnav"
EVAL_FREQ = 10000
N_EVAL_EPISODES = 20

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#Environment 
train_env = make_vec_env(ISSNavEnv, n_envs=N_ENVS)
eval_env = ISSNavEnv()

#Callbacks
eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_DIR,
                             log_path=LOG_DIR, eval_freq=EVAL_FREQ,
                             n_eval_episodes=N_EVAL_EPISODES, deterministic=True, render=False, verbose=1)

checkpoint_callback = CheckpointCallback(save_freq=EVAL_FREQ, save_path=MODEL_DIR, name_prefix=MODEL_NAME, verbose=1)

#Model
model = PPO(policy="MlpPolicy", env=train_env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]),tensorboard_log=LOG_DIR, verbose=1)

#Train
print("Training PPO agent on ISSNavEnv")
print(f"Total timesteps: {TOTAL_TIMESTEPS}, Parallel envs: {N_ENVS}")
print(f"Network architecture: [64, 64]")

start = time.time()
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_callback, checkpoint_callback], progress_bar=True)
elapsed = time.time() - start

#Save final model
model.save(os.path.join(MODEL_DIR, MODEL_NAME + "_final"))
print(f"Training completed in {elapsed/60:.2f} minutes. Final model saved to {MODEL_DIR}/{MODEL_NAME}_final.zip")

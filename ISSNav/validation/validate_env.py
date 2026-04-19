import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.issnav_v0 import ISSNavEnv
from gymnasium.utils.env_checker import check_env

env = ISSNavEnv()
check_env(env, warn=True)
print("Environment passed the check_env validation!")

#Random agent roll out test 
env = ISSNavEnv()
obs, _ = env.reset(seed=42)
print(f"Initial observation: {obs}")
print(f"Agent position: {env.agent_pos}, Goal position: {env.goal_pos}")

total_reward = 0.0
for step in range(700):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        print(f"Episode ended at step {step+1} with total reward: {total_reward:.2f}")
        break
    if truncated:
        print(f"Episode truncated at step {step+1} with total reward: {total_reward:.2f}")
        break


env.close()
print("Validation complete!")
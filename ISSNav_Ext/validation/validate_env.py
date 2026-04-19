import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gymnasium.utils.env_checker import check_env
from env.issnav_v1 import ISSNavEnvV1

# ---------------------------------------------------------------------------
# Validate ISSNav-v1 (Second-Order Dynamics)
# ---------------------------------------------------------------------------
print("=" * 55)
print("Validating ISSNav-v1 (Second-Order Dynamics)")
print("=" * 55)

env = ISSNavEnvV1()
check_env(env)
print("env_checker passed!\n")

obs, _ = env.reset(seed=42)
print(f"Obs shape  : {obs.shape}  (expected 6)")
print(f"Agent pos  : {env.agent_pos}")
print(f"Velocity   : {env.velocity}  (should be [0. 0.])")
print(f"Goal pos   : {env.goal_pos}")

print("\nRunning random agent rollout...")
total_reward = 0
for step in range(700):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated:
        print(f"Goal reached at step {step+1} | Reward: {total_reward:.2f}")
        break
    if truncated:
        print(f"Truncated at step {step+1} | Reward: {total_reward:.2f}")
        break

env.close()
print("\nValidation complete.")

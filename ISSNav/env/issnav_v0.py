import numpy as np
import gymnasium as gym
from gymnasium import spaces
from maps.iss_map import build_iss_map, MAP_W, MAP_H

class ISSNavEnv(gym.Env):
    """
    ISSNav-v0: 2D goal - conditioned navigation environment
    inspired by the ISS interior layout.

    Observation: (x, y, goal_x, goal_y) normalized to [0, 1]
    Action: (vx, vy) continuous velocity command in [-1, 1]
    Rewards: +1.0 goal reached, -0.20 wall collision, -0.01 per timestep
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, max_steps=700):
        super().__init__()
        self.grid, self.module_centers = build_iss_map()
        self.map_h, self.map_w = self.grid.shape
        self.max_steps = max_steps
        self.goal_threshold = 1.5
        self.render_mode = render_mode
        
        self.action_space = spaces.Box(low = -1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low= 0.0, high = 1.0, shape=(4,), dtype=np.float32)
        
        sc, sr = self.module_centers["nauka"]
        gc, gr = self.module_centers["columbus"]
        self.agent_pos = np.array([sc, sr], dtype=np.float32)
        self.goal_pos  = np.array([gc, gr], dtype=np.float32)
        self.steps = 0
        self.module_names = list(self.module_centers.keys())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        indices = self.np_random.choice(len(self.module_names), size=2, replace=False)

        start = self.module_names[indices[0]]
        goal = self.module_names[indices[1]]

        sc, sr = self.module_centers[start]
        gc, gr = self.module_centers[goal]

        self.agent_pos = np.array([sc, sr], dtype=np.float32)
        self.goal_pos = np.array([gc, gr], dtype=np.float32)
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        action = np.clip(action, -1.0, 1.0)

        new_pos = self.agent_pos + action
        col = int(np.clip(round(new_pos[0]), 0, self.map_w - 1))
        row = int(np.clip(round(new_pos[1]), 0, self.map_h - 1))

        if self.grid[row, col] == 1:
            reward = -0.20
            terminated = False
        
        else:
            self.agent_pos = np.array([
                np.clip(new_pos[0], 0, self.map_w - 1),
                np.clip(new_pos[1], 0, self.map_h - 1)
            ], dtype=np.float32)

            dist = np.linalg.norm(self.agent_pos - self.goal_pos)
            if dist < self.goal_threshold:
                reward = 1.0
                terminated = True
            else:
                reward = -0.01
                terminated = False
        
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _get_obs(self):
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        return np.array([ax / (self.map_w - 1), 
                         ay / (self.map_h - 1),
                         gx / (self.map_w - 1), 
                         gy / (self.map_h - 1)], dtype=np.float32)
    
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO
        import cv2

        fig, ax = plt.subplots(figsize=(14, 5))
        display = np.where(self.grid == 1, 0.15, 0.93)
        ax.imshow(display, cmap='gray', origin='upper',
                  vmin=0, vmax=1, interpolation='nearest')
        
        for name, (col, row) in self.module_centers.items():
            ax.text(col, row, name.replace('_', '\n'),
                    ha='center', va='center',
                    fontsize=6, color='steelblue', fontweight='bold')
        
        ax.plot(self.goal_pos[0], self.goal_pos[1], 'r*', markersize=13, zorder=5)
        ax.plot(self.agent_pos[0], self.agent_pos[1], 'go', markersize=10, zorder=5)
        ax.set_title(f"ISSNav-v0, Step: {self.steps}", fontsize=10)
        ax.axis('off')
        plt.tight_layout(pad=0.3)

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi =80)
        plt.close(fig)
        buf.seek(0)

        img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def close(self):
        pass

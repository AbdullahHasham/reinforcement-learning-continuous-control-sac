import gymnasium as gym
from gymnasium import Wrapper
import numpy as np


class CustomRewardWrapper(Wrapper):
    """
    Custom reward wrapper for MountainCarContinuous-v0.

    Default reward in MountainCarContinuous-v0:
    - reward = 100 if goal reached (position >= 0.45)
    - reward = -0.1 * (action^2) for control cost (penalizes large actions)

    Problem with default reward:
    - Very sparse: agent only gets positive reward at the goal
    - Car must learn to swing back and forth to build momentum
    - Without shaping, agent struggles to discover the solution

    Custom reward design (Potential-Based Reward Shaping):
    Uses F(s,s') = gamma * phi(s') - phi(s) which is proven to preserve
    optimal policies while accelerating learning.

    Potential function phi(s) combines:
    1. Height potential: Rewards being at higher positions (potential energy)
    2. Kinetic energy: Rewards having velocity (ability to climb)
    3. Goal proximity: Extra reward when close to goal
    """

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg
        self.min_position = -1.2
        self.max_position = 0.6
        self.goal_position = 0.45
        self.gamma = 0.99
        self.prev_potential = None

    def _height(self, position):
        """Calculate height at given position (MountainCar terrain)."""
        return np.sin(3 * position) * 0.45 + 0.55

    def _potential(self, position, velocity):
        """
        Calculate potential function for state (position, velocity).
        Higher potential = closer to solving the task.
        """
        height = self._height(position)
        height_potential = height * self.cfg.get('height_weight', 1.0)

        kinetic_energy = 0.5 * velocity ** 2
        kinetic_potential = kinetic_energy * self.cfg.get('velocity_weight', 10.0)

        goal_distance = abs(self.goal_position - position)
        goal_potential = (1.0 - goal_distance / (self.max_position - self.min_position))
        goal_potential = goal_potential * self.cfg.get('goal_proximity_weight', 5.0)

        position_normalized = (position - self.min_position) / (self.max_position - self.min_position)
        position_potential = position_normalized * self.cfg.get('position_weight', 1.0)

        return height_potential + kinetic_potential + goal_potential + position_potential

    def reset(self, **kwargs):
        """Reset environment and initialize potential."""
        obs, info = self.env.reset(**kwargs)
        position, velocity = obs[0], obs[1]
        self.prev_potential = self._potential(position, velocity)
        return obs, info

    def step(self, action):
        """Take step and compute shaped reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        position, velocity = obs[0], obs[1]
        current_potential = self._potential(position, velocity)

        if self.prev_potential is not None:
            shaping_reward = self.gamma * current_potential - self.prev_potential
        else:
            shaping_reward = 0.0

        self.prev_potential = current_potential

        shaped_reward = reward + shaping_reward

        if position >= self.goal_position:
            shaped_reward += 10.0

        return obs, shaped_reward, terminated, truncated, info
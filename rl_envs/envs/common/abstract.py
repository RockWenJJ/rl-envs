import copy
import os
import gym
import numpy as np

from typing import List, Tuple, Optional, Callable
from gym import Wrapper
from gym.utils import seeding

from rl_envs.envs.common.graphics import EnvViewer

class AbstractEnv(gym.Env):
    def __init__(self, config: dict=None) -> None:
        # configuration
        self.config = config

        # Seeding
        self.np_random = None
        self.seed()

        # Scene
        self.road = None
        self.controlled_vehicles = []

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        # self.define_spaces()
        self.object_index = 0

        # Running
        self.time = 0   # Simulation Time
        self.steps = 0  # Actions preformed
        self.done = False

        # Rendering
        self.viewer = None
        self._monitor = None
        self.rendering_mode = 'human'
        self.enable_auto_render = False

        # Env objects
        self.vehicles = []
        self.road = None
        self.frequency = 15

        self._reset()

    def seed(self, seed: int=None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self)->None:
        raise NotImplementedError()

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        '''Render the environment'''
        self.rendering_mode = mode
        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        self.viewer.display()

    def reset(self):
        obs = self._reset()
        return obs

    def configure(self, config):
        self.config = config





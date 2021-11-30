import os
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union
import numpy as np
import pygame

from rl_envs.road.graphics import RoadGraphics
from rl_envs.vehicle.graphics import VehicleGraphics

PositionType = Union[Tuple[float, float], np.ndarray]
#
# class Color:
#     BLACK = (0, 0, 0)
#     GREY = (150, 150, 150)
#     GREEN = (50, 200, 0)
#     YELLOW = (200, 200, 0)
#     WHITE = (255, 255, 255)

class EnvViewer:
    '''A viewer to render a simulated environment'''
    SAVE_IMAGES = False

    def __init__(self, env, config: Optional[dict] = None) -> None:
        self.env = env
        self.config = config or env.config
        self.offscreen_rendering = False

        pygame.init()
        pygame.display.set_caption("Highway-env")
        panel_size = (self.config["screen_width"], self.config["screen_height"])

        if not self.offscreen_rendering:
            self.screen = pygame.display.set_mode([self.config['screen_width'], self.config['screen_height']])

        # setup world sim_surface
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = self.config.get("scaling", self.sim_surface.INITIAL_SCALING)
        self.sim_surface.centering_position = self.config.get("centering_position", self.sim_surface.INITIAL_CENTERING)
        self.clock = pygame.time.Clock()

        self.enabled = True
        self.agent_display = None
        self.agent_surface = None
        self.vehicle_trajectory = None
        self.frame = 0
        self.directory = None


    def handle_events(self) -> None:
        '''Handle pygame events by forwarding them to the display and environment vehicle'''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.action_type: # handle keyboard action input
                EventHandler.handle_event(self.env.action_type, event)

    def display(self) -> None:
        '''Display the road and vehicles on a pygame window'''
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        for v in self.env.vehicles + self.env.controlled_vehicles:
            VehicleGraphics.display(v, self.sim_surface)

        if not self.offscreen_rendering:
            self.screen.blit(self.sim_surface, (0, 0))
            # real time rendering
            self.clock.tick(15) # config simulation_frequency
            pygame.display.flip()


    def window_position(self) -> np.ndarray:
        ''' get the world position of the center of the displayed window'''
        # if self.env.vehicle:
        #     return self.env.vehicle.position
        # else:
        #     return np.array([0, 0])
        return np.array([0, 0])

    def close(self) -> None:
        '''Close the pygame window'''
        pygame.quit()


class WorldSurface(pygame.Surface):
    '''A pygame Surface implementing a local coordinate system so that we can move or zoom in the displayed area'''
    INITIAL_SCALING = 16  # real_world pos x scaling = simulated pixel
    INITIAL_CENTERING = [0.5, 0.5]
    SCALING_FACTOR = 1.3
    MOVING_FACTOR = 0.1

    BLACK = (0, 0, 0)
    GREY = (150, 150, 150)
    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    WHITE = (255, 255, 255)

    def __init__(self, size: Tuple[int, int], flags: object, surf: pygame.SurfaceType) -> None:
        super().__init__(size, flags, surf)
        self.origin = np.array([0, 0])
        self.scaling = self.INITIAL_SCALING
        self.centering_position = self.INITIAL_CENTERING

    def pix(self, length: float) -> int:
        '''Convert a distance [m] to pixels [px]'''
        return int(length * self.scaling)

    def pos2pix(self, x: float, y: float) -> Tuple[int, int]:
        '''Convert world coordinates [m] into a position in the surface [px]'''
        return self.pix(x - self.origin[0]), self.pix(y - self.origin[1])

    def vec2pix(self, vec: PositionType) -> Tuple[int, int]:
        return self.pos2pix(vec[0], vec[1])

    def is_visible(self, vec: PositionType, margin: int = 50) -> bool:
        x, y = self.vec2pix(vec)
        return -margin < x < self.get_width() + margin and -margin < y < self.get_height() + margin

    def move_display_window_to(self, position: PositionType) -> None:
        self.origin = position - np.array(
            [self.centering_position[0] * self.get_width() / self.scaling,
             self.centering_position[1] * self.get_height() / self.scaling])

    def handle_event(self, event: pygame.event.EventType) -> None:
        '''Handle pygame events for moving and zooming in the displayed area'''
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l: # press `l`
                self.scaling *= 1 / self.SCALING_FACTORR
            if event.key == pygame.K_o: # press `o`
                self.scaling *= self.SCALING_FACTOR
            if event.key == pygame.K_m: # press `m`
                self.centering_position[0] -= self.MOVING_FACTOR
            if event.key == pygame.K_k: # press `k`
                self.centering_position[0] += self.MOVING_FACTOR


class EventHandler:
    '''Handle discrete or continuous action acted by the agent'''
    @classmethod
    def handle_event(cls, action_type,  event: pygame.event.EventType) -> None:
        #TODO: implement later
        pass



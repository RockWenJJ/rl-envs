from typing import List, Tuple, TYPE_CHECKING, Union, Sequence

import numpy as np
import pygame
from rl_envs.vehicle.vehicle import ControlledVehicle

Vector = Union[np.ndarray, Sequence[float]]

class VehicleGraphics:
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = BLUE
    EGO_COLOR = GREEN

    @classmethod
    def display(cls, vehicle, surface,
                transparent: bool = False,
                offscreen: bool = False,
                label: bool = False):
        '''Display a vehicle on a pygame surface'''
        if not surface.is_visible(vehicle.position): # return if out of visible region
            return

        v = vehicle
        tire_length = 1.
        length = v.LENGTH + 2
        vehicle_surface = pygame.Surface((surface.pix(length), surface.pix(length)),
                                         flags=pygame.SRCALPHA)
        rect = (surface.pix(tire_length),
                surface.pix(length/2 - v.WIDTH/2),
                surface.pix(v.LENGTH),
                surface.pix(v.WIDTH))

        color = cls.get_color(v, transparent)
        pygame.draw.rect(vehicle_surface, color, rect, 0)
        pygame.draw.rect(vehicle_surface, cls.BLACK, rect, 1)

        # Centered rotation
        h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
        position = np.array([*surface.pos2pix(v.position[0], v.position[1])])

        if not offscreen:
            # convert_alpha throws errors in offscreen mode
            # see https://stackoverflow.com/a/19057853
            vehicle_surface = pygame.Surface.convert_alpha(vehicle_surface)
        cls.blit_rotate(surface, vehicle_surface, position, np.rad2deg(-h))

    @classmethod
    def get_color(cls, vehicle, transparent: bool = False) -> Tuple[int]:
        color = cls.DEFAULT_COLOR
        if getattr(vehicle, "color", None):
            color = vehicle.color
        elif vehicle.crashed:
            color = cls.RED
        elif isinstance(vehicle, ControlledVehicle):
            color = cls.GREEN
        if transparent:
            color = color
            color = (color[0], color[1], color[2], 50)
        return color

    @classmethod
    def blit_rotate(cls, surf: pygame.SurfaceType, image: pygame.SurfaceType, pos: Vector, angle: float,
                    origin_pos: Vector = None, show_rect: bool = False) -> None:
        """Many thanks to https://stackoverflow.com/a/54714144."""
        # calculate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

        # calculate the translation of the pivot
        if origin_pos is None:
            origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (
        pos[0] - origin_pos[0] + min_box[0] - pivot_move[0], pos[1] - origin_pos[1] - max_box[1] + pivot_move[1])
        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        # rotate and blit the image
        surf.blit(rotated_image, origin)
        # draw rectangle around the image
        if show_rect:
            pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()), 2)

from typing import List, Tuple, Union, TYPE_CHECKING

import numpy as np
import pygame
# from rl_envs.road.road import Road
from rl_envs.road.lane import LineType

class LaneGraphics:
    '''A visualization of a lane'''
    STRIPE_SPACING: float = 4.33
    STRIPE_LENGTH: float = 3
    STRIPE_WIDTH: float = 0.3

    @classmethod
    def display(cls, lane, surface) -> None:
        '''Display a lane on a surface'''
        stripes_count = int(2 * (surface.get_height() + surface.get_width())
                            / (cls.STRIPE_SPACING * surface.scaling))  # number of stripes we should draw
        s_origin, _ = lane.local_coord(surface.origin)                # origin's local longitudinal coordinate
        s0 = (int(s_origin) // cls.STRIPE_SPACING - stripes_count // 2) \
             * cls.STRIPE_SPACING # longitudinal of the first stripe
        for side in range(2):
            if lane.line_types[side] == LineType.STRIPED:
                cls.striped_line(lane, surface, stripes_count, s0, side)
            elif lane.line_types[side] == LineType.CONTINUOUS_LINE:
                cls.continuous_line(lane, surface, stripes_count, s0, side)
            elif lane.line_types[side] == LineType.CONTINUOUS:
                cls.continuous_curve(lane, surface, stripes_count, s0, side)

    @classmethod
    def draw_stripes(cls, lane, surface, starts, ends, lats) -> None:
        '''
        Draw lane stripes
        :param lane: the lane to draw
        :param surface: the surface to draw on
        :param starts: a List of starting longitudinal positions for each stripe [m]
        :param ends: a List of ending longitudinal positions for each stripe [m]
        :param lats: a List of lateral positions for each stripe [m]
        :return:
        '''
        starts = np.clip(starts, 0, lane.length)
        ends = np.clip(ends, 0, lane.length)
        for i, _ in enumerate(starts):
            if abs(starts[i] - ends[i]) > 0.5 * cls.STRIPE_LENGTH:
                pygame.draw.line(surface, surface.WHITE,
                                 (surface.vec2pix(lane.position(starts[i], lats[i]))),
                                 (surface.vec2pix(lane.position(ends[i], lats[i]))),
                                 max(surface.pix(cls.STRIPE_WIDTH), 1))

    @classmethod
    def striped_line(cls, lane, surface, stripes_count, longitudinal, side) -> None:
        starts = longitudinal + np.arange(stripes_count) * cls.STRIPE_SPACING
        ends = longitudinal + np.arange(stripes_count) * cls.STRIPE_SPACING + cls.STRIPE_LENGTH
        lats = [(side - 0.5) * lane.width_at(s) for s in starts] # side 0: left, 1: right
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def continuous_line(cls, lane, surface, stripes_count, longitudinal, side) -> None:
        starts = [longitudinal + 0 * cls.STRIPE_SPACING]
        ends = [longitudinal + stripes_count * cls.STRIPE_SPACING + cls.STRIPE_LENGTH]
        lats = [(side - 0.5) * lane.width_at(s) for s in starts] # side 0: left, 1: right
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def continuous_curve(cls, lane, surface, stripes_count, longitudinal, side) -> None:
        cls.continuous_line(lane, surface, stripes_count, longitudinal, side)

    @classmethod
    def draw_ground(cls, lane, surface, color, width,
                    draw_surface: pygame.Surface = None) -> None:
        draw_surface = draw_surface or surface
        stripes_count = int(2 * (surface.get_height() + surface.get_width()) / (cls.STRIPE_SPACING * surface.scaling))
        s_origin, _ = lane.local_coordinates(surface.origin)
        s0 = (int(s_origin) // cls.STRIPE_SPACING - stripes_count // 2) * cls.STRIPE_SPACING
        dots = []
        for side in range(2):
            longis = np.clip(s0 + np.arange(stripes_count) * cls.STRIPE_SPACING, 0, lane.length)
            lats = [2 * (side - 0.5) * width for _ in longis]
            new_dots = [surface.vec2pix(lane.position(longi, lat)) for longi, lat in zip(longis, lats)]
            new_dots = reversed(new_dots) if side else new_dots
            dots.extend(new_dots)
        pygame.draw.polygon(draw_surface, color, dots, 0)


class RoadGraphics:
    '''A visualization of a road lanes and vehicles'''

    @staticmethod
    def display(road,  surface) -> None:
        '''Display the road lanes on a world surface'''
        surface.fill(surface.GREY)
        for _from in road.lanes_graph.keys():
            for _to in road.lanes_graph[_from].keys():
                for lane in road.lanes_graph[_from][_to]:
                    LaneGraphics.display(lane, surface)


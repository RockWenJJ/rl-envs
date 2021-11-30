from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Optional
import numpy as np

from rl_envs.utils import *

class LineType:
    '''Lane side line type'''
    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3

class AbstractLane:

    VEHICLE_LENGTH: float = 5
    DEFAULT_WIDTH: float = 5
    length: float = 0
    @abstractmethod
    def position(self, longitudinal:float, lateral:float)->np.ndarray:
        '''Convert local lane coordinates to world position'''
        raise NotImplementedError

    @abstractmethod
    def local_coord(self, position:np.ndarray)->Tuple[float, float]:
        '''Convert world position to local lane coordinates'''
        raise NotImplementedError

    @abstractmethod
    def heading_at(self, longitudinal:float)->float:
        '''Get the lane heading at a given longitudinal lane position'''
        raise NotImplementedError

    @abstractmethod
    def width_at(self, longitudinal:float)->float:
        '''Get the lane width at a given longitunial lane coordinate'''
        raise NotImplementedError

    def on_lane(self, position:np.ndarray, margin: float=0)->bool:
        '''Given a world position, check whether it is on the lane'''
        longitudinal, lateral = self.local_coord(position)
        is_on = np.abs(lateral) <= self.width_at(longitudinal) / 2 + margin and \
                -self.VEHICLE_LENGTH <= longitudinal < self.length + self.VEHICLE_LENGTH
        return is_on

    def is_reachable_from(self, position: np.ndarray)->bool:
        '''Whether the lane is reachable from a given world position'''
        #TODO:
        pass

    def distance(self, position: np.ndarray):
        '''Compute the L1 distance from a given position to lane.'''
        s, r = self.local_coord(position)
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0)

    def distance_with_heading(self, position: np.ndarray, heading: Optional[float],
                              heading_weight: float = 1.0):
        '''Compute a weighted distance in position and heading to the lane.'''
        if heading is None:
            return self.distance(position)
        s, r = self.local_coord(position)
        angle = np.abs(wrap_to_pi(heading - self.heading_at(s)))
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0) + heading_weight * angle

class StraightLane(AbstractLane):
    '''A lane going in straight line'''
    def __init__(self,
                 start: np.ndarray,
                 end: np.ndarray,
                 width: float = AbstractLane.DEFAULT_WIDTH,
                 line_types: Tuple[LineType, LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = 30,
                 priority: int = 0) -> None:
        '''
        :param start: lane starting position <x, y>[m]
        :param end: lane ending position <x, y> [m]
        :param width: the lane width
        :param line_types: the type of lines on both sides of the lane [left, right]
        :param forbidden: is changing to this lane forbidden
        :param speed_limit: maximum speed allowed on the lane
        :param priority: priority level of the lane, determining who has the right of way
        '''
        self.start, self.end = start, end
        self.width = width
        self.heading = np.arctan2(self.end[1] - self.start[1], self.end[0]-self.start[0])
        self.length = np.linalg.norm(self.end - self.start)
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.direction = (self.end - self.start) / self.length # [cos, sin]
        self.direction_lateral = np.array([-self.direction[1], self.direction[0]])
        self.forbidden = forbidden
        self.priority = priority
        self.speed_limit = speed_limit

    def position(self, longitudinal:float, lateral:float)->np.ndarray:
        '''Convert local lane coordinates to world position'''
        position = self.start + longitudinal * self.direction + lateral * self.direction_lateral
        return position

    def local_coord(self, position:np.ndarray)->Tuple[float, float]:
        '''Convert world position to local lane coordinates'''
        delta = position - self.start
        longitudinal = np.dot(delta, self.direction)
        lateral = np.dot(delta, self.direction_lateral)
        return float(longitudinal), float(lateral)

    def heading_at(self, longitudinal:float)->float:
        '''Get the lane heading at a given longitudinal lane position'''
        return self.heading # Note that it is a straight lane

    def width_at(self, longitudinal: float) -> float:
        '''Get the lane width at a given longitunial lane coordinate'''
        return self.width # Note that it is a straight lane

class CircularLane(AbstractLane):

    """A lane going in circle arc."""

    def __init__(self,
                 center,
                 radius: float,
                 start_phase: float,
                 end_phase: float,
                 clockwise: bool = True,
                 width: float = AbstractLane.DEFAULT_WIDTH,
                 line_types: List[LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:
        super().__init__()
        self.center = np.array(center)
        self.radius = radius
        self.start_phase = start_phase
        self.end_phase = end_phase
        self.direction = 1 if clockwise else -1
        self.width = width
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.forbidden = forbidden
        self.length = radius*(end_phase - start_phase) * self.direction
        self.priority = priority
        self.speed_limit = speed_limit

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        return self.center + (self.radius - lateral * self.direction)*np.array([np.cos(phi), np.sin(phi)])

    def heading_at(self, longitudinal: float) -> float:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        psi = phi + np.pi/2 * self.direction
        return psi

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coord(self, position: np.ndarray) -> Tuple[float, float]:
        delta = position - self.center
        phi = np.arctan2(delta[1], delta[0])
        phi = self.start_phase + wrap_to_pi(phi - self.start_phase)
        r = np.linalg.norm(delta)
        longitudinal = self.direction*(phi - self.start_phase)*self.radius
        lateral = self.direction*(self.radius - r)
        return longitudinal, lateral


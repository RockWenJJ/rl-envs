import numpy as np
import logging
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional

from rl_envs.road.lane import LineType, StraightLane, AbstractLane

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]

class RoadNetwork:
    lanes_graph: Dict[str, Dict[str, List[AbstractLane]]] # graph that contains all lanes

    def __init__(self):
        self.np_random = np.random.RandomState()
        self.lanes_graph = {}

    def add_lane(self, _from: str, _to: str, lane: AbstractLane)->None:
        '''A lane is encoded as an edge in the road network'''
        if _from not in self.lanes_graph:
            self.lanes_graph[_from] = {}
        if _to not in self.lanes_graph[_from]:
            self.lanes_graph[_from][_to] = []
        self.lanes_graph[_from][_to].append(lane)

    def get_lane(self, index: LaneIndex)->AbstractLane:
        ''' Get the lane corresponding to a given index in the road network'''
        _from, _to, _id = index
        if _id is None and len(self.lanes_graph[_from][_to]) == 1:
            _id = 0
        return self.lanes_graph[_from][_to][_id]

    def get_closest_lane_index(self, position: np.ndarray, heading: Optional[float]=None)->LaneIndex:
        '''Get the index of the lane closest to a world position'''
        #TODO: implement this function
        indexes, distances = [], []
        for _from, to_dict in self.lanes_graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def get_lane_index(self, lane):
        for _from, to_dict in self.lanes_graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    if l == lane:
                        return (_from, _to, _id)


    def add_intersection(self, intersection: "Intersection"):
        pass

class StraightRoadNetwork(RoadNetwork):
    def __init__(self,
                 lanes: int = 4,
                 start: float = 0,
                 length: float = 1000,
                 angle: float = 0,
                 speed_limit: float = 30,
                 nodes_str: Optional[Tuple[str, str]] = None)->None:
        super().__init__()
        nodes_str = nodes_str or ("0", "1")
        for lane in range(lanes):
            origin = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([start + length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            origin = rotation @ origin
            end = rotation @ end
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE]
            self.add_lane(*nodes_str, StraightLane(origin, end, line_types=line_types, speed_limit=speed_limit))

class Intersection:
    def __init__(self, corners: np.ndarray,
                 node_names: List[str],
                 node_positions: np.ndarray,
                 node_headings: np.ndarray)->None:
        '''

        :param corners: corner positions of the intersection, starts from left-up => right-up => right-down => left-down
        :param node_names: all the input and output node names
        :param node_positions: all the input and output node positions
        :param node_headings: all the input and output node headings
        '''
        self.corners = corners
        self.node_names = node_names
        self.node_positions = node_positions
        self.node_headings = node_headings

    def plan_route(self, start, end):
        pass










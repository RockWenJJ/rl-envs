from typing import Dict, Tuple
from gym.envs.registration import register
import numpy as np
from copy import deepcopy

from rl_envs.envs.common.abstract import AbstractEnv
from rl_envs.road.road import RoadNetwork
from rl_envs.road.lane import LineType, AbstractLane, StraightLane, CircularLane
from rl_envs.vehicle.vehicle import Vehicle, ControlledVehicle

class IntersectionEnv(AbstractEnv):
    def __init__(self, config:dict=None)->None:
        super().__init__(config)

    def _has_crashed(self):
        return any(vehicle.crashed() for vehicle in self.controlled_vehicles)

    def _has_arrived(self):
        return all(vehicle.arrived() for vehicle in self.controlled_vehicles)

    def _has_offroad(self):
        return any(vehicle.offroad() for vehicle in self.controlled_vehicles)

    def _reset(self) -> None:
        self.object_index = 0
        self._make_road()
        self._make_vehicles()
        self.steps = 0
        # for simple situations with only ego vehicle,
        obs = self._observe()
        return obs

    def _make_road(self) -> None:
        '''Make an 4-way intersection
                 |  ini3 |  outo3 |
                 |       |        |
        _________| ino3  |  outi3 |_________
        outo2 outi2               ino0    ini0
        ---------                 ---------
        in2    ino2               outi0   outo0
        ---------| outi1 |  ino1 |---------
                 |       |       |
                 | outo1 |  in1  |
        The horizontal lane has the right of way. The levels of priority are:
        - 3 : horizontal straight lanes and right-turns
        - 2 : vertical straight lanes and right-turns
        - 1 : horizontal left-turns
        - 0 : vertical left-turns

        '''
        lane_width = AbstractLane.DEFAULT_WIDTH
        roadnet = RoadNetwork()

        # make lanes
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        node_names = []
        for corner in range(4):  # corner runs in clockwise!!
            angle = np.radians(90 * corner)
            is_horizontal = not corner % 2
            priority = 3 if is_horizontal else 1 #TODO: check how priority used in this environment
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            # out
            start = rotation @ np.array([lane_width, lane_width/2])
            end = rotation @ np.array([lane_width + 20,  lane_width/2])
            roadnet.add_lane("outi"+str(corner), "outo"+str(corner),
                             StraightLane(start, end, line_types=[s, c], speed_limit=20, priority=priority))

            # income
            start = rotation @ np.array([lane_width + 20, -lane_width/2])
            end = rotation @ np.array([lane_width, -lane_width/2])
            roadnet.add_lane("ini" + str(corner), "ino" + str(corner),
                             StraightLane(start, end, line_types=[s, c], speed_limit=20, priority=priority))

            # left turn at intersection
            l_center = rotation @ np.array([lane_width, lane_width])
            l_radius = lane_width + lane_width / 2.
            roadnet.add_lane("ino"+str(corner), "outi"+str((corner+1)%4),
                             CircularLane(l_center, l_radius, angle + np.radians(-90), angle + np.radians(-180),
                                          clockwise=False, line_types=[n, n], priority=priority-1, speed_limit=15.))

            # right turn at intersection
            r_center = rotation @ np.array([lane_width, -lane_width])
            r_radius = lane_width / 2
            roadnet.add_lane("ino"+str(corner), "outi"+str((corner-1)%4),
                             CircularLane(r_center, r_radius, angle + np.radians(90), angle + np.radians(180),
                                          clockwise=True, line_types=[n, n], priority=priority, speed_limit=15.))

            # straight forward at intersection
            start = rotation @ np.array([lane_width, -lane_width/2]) # same with income's end
            end = rotation @ np.array([-lane_width, -lane_width/2])
            roadnet.add_lane("ino"+str(corner), "outi"+str((corner+2)%4),
                             StraightLane(start, end, line_types=[n, n], speed_limit=20, priority=priority))


        self.road = roadnet

    def _make_vehicles(self):
        self.vehicles = []
        self.controlled_vehicles = []
        # self.vehicles.append(Vehicle.create_random(self.road))

        # create controlled vehicle
        _from = 'ini1'
        _to = 'ino1'
        _id = 0
        lane = self.road.get_lane((_from, _to, _id))
        speed = 9.0
        ego_v = ControlledVehicle(0, self.road, lane, lane.position(0, 0), lane.heading_at(0), speed)
        ego_v.set_destination(self.road, 'outo2')
        self.controlled_vehicles.append(ego_v)

        # _from = 'ini0'
        # _to = 'ino0'
        # _id = 0
        # lane = self.road.get_lane((_from, _to, _id))
        # speed = 11
        # v = Vehicle(1, self.road, lane, lane.position(0, 0), lane.heading_at(0), speed)
        # v.set_destination(self.road, 'outo2')
        # self.vehicles.append(v)

        # _from = 'ini0'
        # _to = 'ino0'
        # _id = 0
        # lane = self.road.get_lane((_from, _to, _id))
        # speed = 10.0
        # v = Vehicle(2, self.road, lane, lane.position(6, 0), lane.heading_at(0), speed)
        # v.set_destination(self.road, 'outo2')
        # self.vehicles.append(v)
        #
        # _from = 'ini3'
        # _to = 'ino3'
        # _id = 0
        # lane = self.road.get_lane((_from, _to, _id))
        # speed = 10.0
        # v = Vehicle(3, self.road, lane, lane.position(6, 0), lane.heading_at(0), speed)
        # v.set_destination(self.road, 'outo1')
        # self.vehicles.append(v)
        #
        # _from = 'ini2'
        # _to = 'ino2'
        # _id = 0
        # lane = self.road.get_lane((_from, _to, _id))
        # speed = 10.0
        # v = Vehicle(4, self.road, lane, lane.position(6, 0), lane.heading_at(0), speed)
        # v.set_destination(self.road, 'outo0')
        # self.vehicles.append(v)

    def _reward(self, action: float=0) -> float:

        return 0.0

    def step(self, action):
        '''Perform action and step the environment dynamics'''
        self.steps += 1
        # for i, v in enumerate(self.vehicles):
        #     v.check_nearby_vehicles(self.vehicles, 1./self.frequency)
        #     v.step(action, 1./self.frequency) # action is useless for

        reward = 0
        for i, v in enumerate(self.controlled_vehicles):
            v.step(action, 1./self.frequency)
            r, info = v.reward()
            reward += r

        obs = self._observe()
        done = info["reach_dest"] or info["off_road"]

        return obs, reward, done, info

    def _observe(self):
        obs = []
        for v in self.controlled_vehicles:
            obs.append(v.observe())
        return deepcopy(obs)


register(id='intersection-env-v0',
         entry_point='rl_envs.envs:IntersectionEnv')
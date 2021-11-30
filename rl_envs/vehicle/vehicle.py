from typing import Union, Optional, Tuple, List
import numpy as np
import copy
from collections import deque
from rl_envs.utils import *

class Vehicle:
    LENGTH: float = 3.5      # vehicle length
    WIDTH: float = 1.5       # vehicle width
    MAX_SPEED: float = 40.   # maximum reachable speed
    MAX_STEERING_ANGLE: float = np.pi/3 # [rad]
    MAX_ACCELERATION: float = 10.0 # [m/s^2]
    DEFAULT_SPEEDS = [8, 15]
    HISTORY_SIZE: int = 30 # length of the vehicle state history
    PREDICTION_SIZE: int = 30

    TAU_ACC: float = 0.5 # [s]
    TAU_HEADING: float = 0.2 # [s]
    TAU_LATERAL: float = 0.5 # [s]
    TAU_PURSUIT: float = 0.5 * TAU_HEADING # [s]

    KP_A: float = 1 / TAU_ACC
    KP_LATERAL: float = 1 / TAU_LATERAL
    KP_HEADING: float = 1/ TAU_HEADING

    ACCELERATION_PARAMETERS = [0.3, 0.3, 2.0]
    DISTANCE_WANTED = 5.0 + LENGTH
    TIME_WANTED = 1.0

    def __init__(self,
                 index, road,
                 lane, position,
                 heading: float = 0,
                 speed: float = 0):
        self.index = index
        self.road = road # the roadnetwork
        self.position = np.array(position, dtype=np.float)
        self.heading = heading
        self.speed = speed
        self.lane_index = self.road.get_lane_index(lane) #self.road.get_closest_lane_index(self.position, self.heading)
        self.lane = lane # self.road.get_lane(self.lane_index) if self.road else None
        self.target_lane_index = self.lane_index


        self.solid = True
        self.check_collisions = True
        self.crashed = False

        self.action = {'steering': 0, 'acceleration': 0}
        self.history = deque(maxlen=self.HISTORY_SIZE)
        self.predictions = deque(maxlen=self.PREDICTION_SIZE)

        self.destination = None
        self.reach_dest = False
        self.nearby_vehicles = []


    @classmethod
    def create_random(cls, road: "RoadNetwork",
                      speed: float = None,
                      lane_from: Optional[str] = None,
                      lane_to: Optional[str] = None,
                      lane_id: Optional[str] = None,
                      spacing: float = 1) -> "Vehicle":
        '''Randomly Create a vehicle on the road'''
        _from = lane_from or road.np_random.choice(list(road.lanes_graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.lanes_graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.lanes_graph[_from][_to]))
        lane = road.get_lane((_from, _to, _id))
        if speed is None:
            speed = road.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1])
        # default_spacing = 12 + 1.0 * speed
        offset = np.random.uniform(0, lane.length)
        # x0 = np.max([lane.local_coord(v.position)[0] for v in ])
        x0 = offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction

    def polygon(self) -> np.ndarray:
        points = np.array([
            [-self.LENGTH / 2, -self.WIDTH / 2],
            [-self.LENGTH / 2, +self.WIDTH / 2],
            [+self.LENGTH / 2, +self.WIDTH / 2],
            [+self.LENGTH / 2, -self.WIDTH / 2],
        ]).T
        c, s = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([
            [c, -s],
            [s, c]
        ])
        points = (rotation @ points).T + np.tile(self.position, (4, 1))
        return np.vstack([points, points[0:1]])

    def set_destination(self, road, dest: str) -> None:
        '''Should be called just after the vehicle be created'''
        if not dest.startswith('outo'):
            print("Destination of {} is not correct (should be `oo`s)".format(self.index))
        cur_from, cur_to, cur_id = self.lane_index
        route = []
        for _from in road.lanes_graph.keys(): # find the target lane
            if _from.startswith("outi"):
                for _to in road.lanes_graph[_from].keys():
                    if _to == dest:
                        target_lane = road.lanes_graph[_from][_to][0]
                        route.append((_from, _to, target_lane))
                        self.dest_pos = target_lane.position(target_lane.length, 0)

        for _from in road.lanes_graph.keys(): # find the intersection lane
            if _from == cur_to:
                target_in = route[-1][0]
                for _to in road.lanes_graph[_from].keys():
                    if _to == target_in:
                        inter_lane = road.lanes_graph[_from][_to][0]
                        route.append((_from, _to, inter_lane))

        route.append((cur_from, cur_to, self.lane))  # ge current lane
        route.reverse()

        self.route = route
        self.destination = dest


    def steering_control(self, target_lane_index) -> float:
        '''Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proporional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        '''
        target_lane = self.road.get_lane(target_lane_index)
        lane_coords = target_lane.local_coord(self.position)
        lane_next_longi = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_longi)

        # lateral position control
        lateral_speed_command = -self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(np.clip(lateral_speed_command / not_zero(self.speed), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # heading control
        heading_rate_command = self.KP_HEADING * wrap_to_pi(heading_ref - self.heading)
        # heading rate to steering angle
        steering_angle = np.arcsin(np.clip(self.LENGTH / 2 / not_zero(self.speed) * heading_rate_command, -1, 1))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        '''Control the speed of the vehicle'''
        return self.KP_A * (target_speed - self.speed)

    def get_target_lane(self):
        longi, _ = self.lane.local_coord(self.position)
        if longi > self.lane.length - self.LENGTH / 2.:
            cur_from, cur_to, cur_idx = self.lane_index
            if cur_to == self.destination:
                self.reach_dest = True
                return self.lane_index, self.lane
            for l_from, l_to, l_next in self.route:
                if l_from == cur_to:
                    return (l_from, l_to, 0), l_next
        else:
            return self.lane_index, self.lane

    def act(self):
        '''
        Perform a high-level action.
        :param action:
        :return:
        '''
        self.target_lane_index, self.target_lane = self.get_target_lane()

        random_acc = (np.random.random() - 0.5) * self.MAX_ACCELERATION
        self.target_speed = max(min(self.speed + random_acc, 15.0), 0)

        # Temporally perform constant speed
        action = {"steering": self.steering_control(self.target_lane_index),
                  "acceleration": self.speed_control(self.target_speed)}

        if len(self.nearby_vehicles) > 0:
            self.nearby_vehicles.sort()
            action["acceleration"] = self.acceleration(self, self.nearby_vehicles[0][1])
            # closest_d = self.nearby_vehicles[0][0]
            # accel = (max(closest_d - self.LENGTH - self.WIDTH, 0.1) - self.speed) * 2
            # action["acceleration"] = accel

        action["acceleration"] = np.clip(action["acceleration"],
                                         -self.MAX_ACCELERATION, self.MAX_ACCELERATION)
        self.action = action

    def step(self, action, dt: float) -> None:
        '''Propagate the vehicle state given its actions'''
        self.act()

        delta_f = self.action["steering"]
        beta = np.arctan(1 / 2 * np.tan(delta_f))

        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action["acceleration"] * dt

        v = self.speed * np.array([np.cos(self.heading + beta),
                                   np.sin(self.heading + beta)])
        self.position += v * dt

        self.on_state_update()

    def on_state_update(self):
        self.lane_index, self.lane = self.get_target_lane()

    def check_nearby_vehicles(self, vehicles, dt):
        '''

        :param vehicles:
        :return: preceeding vehicles list: (distance, vehicle)
        '''
        self.nearby_vehicles = []
        # just check current lane and next lane
        next_lane = []
        _, cur_to, _ = self.lane_index
        for l_from, l_to, l_next in self.route:
            if l_from == cur_to:
                next_lane.append(l_next)

        s, l = self.lane.local_coord(self.position)
        for i, v in enumerate(vehicles):
            if v.index != self.index:
                d = v.position - self.position
                angle = np.arctan2(d[1], d[0])
                delta = wrap_to_pi(angle - self.heading)
                # introduce some randomness to avoid same distance
                distance = abs(np.linalg.norm(d) * np.cos(delta)) + np.random.random() * 0.1
                v_p = v.lane.priority
                if abs(delta) < np.pi/3 and self.WIDTH < distance < self.speed * 1.0 and v_p >= self.lane.priority:
                    self.nearby_vehicles.append((distance, v))

        # for i, v in enumerate(vehicles):
        #     if v.index != self.index:
        #         v_pos_fut = v.position + v.speed * dt
        #         if self.lane.on_lane(v_pos_fut):
        #             s0_v, l0_v = self.lane.local_coord(v_pos_fut)
        #             if 0 < s0_v - s < self.speed * 1.0 and v.lane.priority >= self.lane.priority:
        #                 distance = s0_v - s
        #                 self.nearby_vehicles.append((distance, v))
        #         elif len(next_lane) > 0 and next_lane[0].on_lane(v_pos_fut):
        #             s1_v, l1_v = next_lane[0].local_coord(v_pos_fut)
        #             if 0 < s1_v + self.lane.length - s < self.speed * 1.0 and v.lane.priority >= self.lane.priority:
        #                 distance = s1_v + self.lane.length -s
        #                 self.nearby_vehicles.append((distance, v))

    def acceleration(self, ego_vehicle, front_vehicle):
        return float(np.dot(self.ACCELERATION_PARAMETERS,
                            self.acceleration_features(ego_vehicle, front_vehicle)))

    def acceleration_features(self, ego_vehicle,
                              front_vehicle) -> np.ndarray:
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = ego_vehicle.target_speed - ego_vehicle.speed
            d_safe = self.DISTANCE_WANTED + np.maximum(ego_vehicle.speed, 0) * self.TIME_WANTED
            if front_vehicle:
                d = np.linalg.norm(ego_vehicle.position - front_vehicle.position)
                dv = min(front_vehicle.speed - ego_vehicle.speed, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])


class ControlledVehicle(Vehicle):
    def __init__(self,index, road,
                 lane, position,
                 heading: float = 0,
                 speed: float = 0):
        super().__init__(index, road, lane, position, heading, speed)

    def step(self, action, dt: float) -> None:
        '''Propagate the vehicle state given its actions'''
        self.action = action

        delta_f = self.action["steering"]
        beta = np.arctan(1 / 2 * np.tan(delta_f))

        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action["acceleration"] * dt

        v = self.speed * np.array([np.cos(self.heading + beta),
                                   np.sin(self.heading + beta)])
        self.position += v * dt

        self.on_state_update()

    def observe(self):
        return [self.position, self.dest_pos]

    def reward(self):
        dest_reward, dist_reward, td_reward, off_road_reward = 0., 0., -0.5, 0.
        info = {"reach_dest": False,
                "off_road": False}
        if self.reach_dest:
            dest_reward = 150
            info["reach_dest"] = True
            return dest_reward, info
        else:
            on_road = False
            for _, _, lane in self.route:
                if lane.on_lane(self.position):
                    on_road = True
            if not on_road:
                off_road_reward = -5.
                info["off_road"] = True
                distance = np.linalg.norm(self.position - self.dest_pos)
                distance = distance if distance > self.LENGTH else 0
                dist_reward  = -np.exp(distance/10)
            reward = td_reward + off_road_reward + dist_reward
            return reward, info















































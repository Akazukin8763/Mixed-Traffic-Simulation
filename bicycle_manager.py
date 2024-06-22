from __future__ import annotations

from typing import TYPE_CHECKING, List

import carla
import numpy as np

import vector
from config import *
from road_user import RoadUser, UserID
from utils import *
from world import World

if TYPE_CHECKING:
    from traffic_manager import TrafficManager


class BicycleManager():
    FIXED_HEIGHT = 0.2

    class BicycleData():
        def __init__(self, init_x, init_y, init_yaw, init_lane, target_x, target_y, forward_x, forward_y):
            self.init_x = init_x
            self.init_y = init_y
            self.init_yaw = init_yaw
            self.init_lane = init_lane
            self.target_x = target_x
            self.target_y = target_y
            self.forward_x = forward_x
            self.forward_y = forward_y
            
    def __init__(self, traffic_manager: TrafficManager, world: World):
        self.manager: TrafficManager = traffic_manager

        self.world: World = world
        self.delta_time: float = self.world.delta_time
        
        self.bicycles: List[carla.Actor] = []  # carla.Agent
        self.num_bicycle: int = 0

        self.positions: np.ndarray = None  # np.ndarray: [[x, y], [x, y], ...]

        self.velocity: np.ndarray = None  # np.ndarray: [[x, y], [x, y], ...]
        self.desired_speed: np.ndarray = None  # np.ndarray: [speed, speed, ...]

        self.com_decel: List[float]= None  # comfortable deceleration, np.ndarray: [decel, decel, ...]
        self.max_accel: List[float] = None  # maximum acceleration, np.ndarray: [accel, accel, ...]
        self.space_headway: List[float] = None  # np.ndarray: [space, space, ...]

        self.forward_vector: List[carla.Vector3D] = None  # list: [carla.Vector3D, carla.Vector3D, ...]
        self.forward_vector_np: np.ndarray = None  # np.ndarray: [[x, y], [x, y], ...]

        self.targets: np.ndarray = None  # np.ndarray: [[x, y], [x, y], ...]

        self.lane_index: List[int] = None  # list: [lane, lane, ...]
        
        self.users: List[UserID] = []  # list: [UserID, UserID, ...]

        self.half_length: List[float] = None
        self.half_width: List[float] = None

        # Data
        self.bicycle_spawn_data: List[BicycleManager.BicycleData] = []  # BicycleData
        self.__read_bicycle_data()

    def __read_bicycle_data(self):
        bicycle_data_path = 'configs/bicycle.csv'
        bicycle_data_rows = read_csv(bicycle_data_path)

        for row in bicycle_data_rows:
            self.bicycle_spawn_data.append(BicycleManager.BicycleData(
                init_x=float(row[0]), 
                init_y=float(row[1]), 
                init_yaw=float(row[2]), 
                init_lane=int(row[3]), 
                target_x=float(row[4]), 
                target_y=float(row[5]),
                forward_x=float(row[6]), 
                forward_y=float(row[7]), 
            ))
    
    def spawn_agent(self) -> bool:
        blueprint = np.random.choice(self.world.blueprintsBicycles)
        spawn_data = np.random.choice(self.bicycle_spawn_data)

        spawn_position = carla.Location(spawn_data.init_x, spawn_data.init_y, BicycleManager.FIXED_HEIGHT)
        spawn_rotation = carla.Rotation(0, spawn_data.init_yaw, 0)
        
        bicycle = self.world.carla_world.try_spawn_actor(
            blueprint, carla.Transform(spawn_position, spawn_rotation)
        )

        if bicycle is not None:
            bicycle.set_simulate_physics(False)
            self.bicycles.append(bicycle)

            if self.num_bicycle == 0:
                self.positions = np.array([[spawn_data.init_x, spawn_data.init_y]])

                self.velocity = np.array([[0.0, 0.0]])
                self.desired_speed = np.array([np.random.normal(Config.Bicycle_desired_speed_mean, Config.Bicycle_desired_speed_std)])

                self.com_decel = [np.random.normal(Config.Bicycle_com_decel_mean, Config.Bicycle_com_decel_std)]
                self.max_accel = [np.random.normal(Config.Bicycle_min_accel_mean, Config.Bicycle_min_accel_std)]
                self.space_headway = [np.random.normal(Config.Bicycle_space_headway_mean, Config.Bicycle_space_headway_std)]

                self.forward_vector = [carla.Vector3D(spawn_data.forward_x, spawn_data.forward_y, 0)]
                self.forward_vector_np = np.array([[spawn_data.forward_x, spawn_data.forward_y]])

                self.targets = np.array([[spawn_data.target_x, spawn_data.target_y]])
                
                self.lane_index = [spawn_data.init_lane]

                half_length, half_width = 1.1089935302734375, 0.433537956882649
                self.half_length = [half_length]
                self.half_width = [half_width]
            else:
                self.positions = np.vstack((self.positions, [spawn_data.init_x, spawn_data.init_y]))

                self.velocity = np.vstack((self.velocity, [0.0, 0.0]))
                self.desired_speed = np.hstack((self.desired_speed, np.random.normal(Config.Bicycle_desired_speed_mean, Config.Bicycle_desired_speed_std)))

                self.com_decel.append(np.random.normal(Config.Bicycle_com_decel_mean, Config.Bicycle_com_decel_std))
                self.max_accel.append(np.random.normal(Config.Bicycle_min_accel_mean, Config.Bicycle_min_accel_std))
                self.space_headway.append(np.random.normal(Config.Bicycle_space_headway_mean, Config.Bicycle_space_headway_std))

                self.forward_vector.append(carla.Vector3D(spawn_data.forward_x, spawn_data.forward_y, 0))
                self.forward_vector_np = np.vstack((self.forward_vector_np, [spawn_data.forward_x, spawn_data.forward_y]))

                self.targets = np.vstack((self.targets, [spawn_data.target_x, spawn_data.target_y]))
                
                self.lane_index.append(spawn_data.init_lane)

                half_length, half_width = 1.1089935302734375, 0.433537956882649
                self.half_length.append(half_length)
                self.half_width.append(half_width)
        
            user = UserID(RoadUser.Bicycle, self.num_bicycle)  # ID
            self.users.append(user)
            self.manager.lanes_road_users[spawn_data.init_lane].append(user)
            self.num_bicycle += 1

            return True
        return False
    
    def step(self):
        if self.num_bicycle == 0:
            return
        
        for idx, bicycle in enumerate(self.bicycles):
            if not bicycle.is_alive:
                continue
            
            force = self.calculate_leader_force_from_current_lane(idx)

            # Update velocity
            self.velocity[idx] += force * self.delta_time

        # Calculate other force
        force_collision_avoidance = self.calculate_repulsive_force_collision_avoidance()
        force_overtaking = self.calculate_overtaking_force(force_collision_avoidance)

        force = self.calculate_driving_force() + \
                self.calculate_repulsive_force_from_lane(self.manager.lanes_line) + \
                force_collision_avoidance + force_overtaking

        # Update velocity
        self.velocity += force * self.delta_time
        self.constrain_velocity()

        # Update position
        self.positions += self.velocity * self.delta_time

        # Set the bicycle transform in simulator
        self.update_transform()

    def update_transform(self):
        for idx, bicycle in enumerate(self.bicycles):
            if not bicycle.is_alive:
                continue

            current_pos = carla.Location(bicycle.get_location().x, bicycle.get_location().y, BicycleManager.FIXED_HEIGHT)
            future_pos = carla.Location(self.positions[idx][0], self.positions[idx][1], BicycleManager.FIXED_HEIGHT)
            target_pos = carla.Location(self.targets[idx][0], self.targets[idx][1], BicycleManager.FIXED_HEIGHT)

            current_to_target_distance = current_pos.distance(target_pos)
            if current_to_target_distance < Config.Epsilon * 5:
                self.positions[idx] = Config.Inf_position
                bicycle.destroy()

                # Remove the bicycle from the lane
                user = self.users[idx]
                for road_users in self.manager.lanes_road_users:
                    if user in road_users:
                        road_users.remove(user)

                continue

            # Set the bicycle transform in simulator
            bicycle.set_location(future_pos)
    
    def constrain_velocity(self):
        # Limit the forward velocity in [0, desired_speed]
        forward_direction = self.forward_vector_np
        forward_speed = np.diagonal(np.dot(self.velocity, forward_direction.T), axis1=0, axis2=1)
        forward_velocity = forward_direction * forward_speed[:, np.newaxis]
        right_velocity = self.velocity - forward_velocity
        right_speed = np.linalg.norm(right_velocity, axis=-1)
        right_direction = right_velocity / right_speed[:, np.newaxis]

        forward_speed = np.clip(forward_speed, np.zeros(self.num_bicycle), self.desired_speed)
        right_speed[forward_speed < Config.Minimum_forward_speed_for_side_movement] = 0

        self.velocity = forward_speed[:, np.newaxis] * forward_direction + right_speed[:, np.newaxis] * right_direction

    def calculate_driving_force(self):
        position = self.positions
        velocity = self.velocity
        target = self.targets

        direction, dist = vector.normalize(target - position)
        force = np.zeros((self.num_bicycle, 2))

        force[dist > Config.Epsilon] = (direction * self.desired_speed.reshape((-1, 1)) - velocity.reshape((-1, 2)))[dist > Config.Epsilon, :]
        force[dist <= Config.Epsilon] = -1.0 * velocity[dist <= Config.Epsilon]
        force /= Config.Tau

        return force
    
    def calculate_repulsive_force_from_lane(self, lanes_line: np.ndarray):
        position = self.positions

        lanes_direction = lanes_line[:, 0] - lanes_line[:, 1]
        to_lanes_start_point = np.expand_dims(position, 1) - lanes_line[:, 0]

        term1 = np.diagonal(np.dot(to_lanes_start_point, lanes_direction.T), axis1=1, axis2=2)
        term2 = np.diagonal(np.dot(lanes_direction, lanes_direction.T), axis1=0, axis2=1)
        projection = lanes_line[:, 0] + np.reshape(term1 / term2, (self.num_bicycle, -1, 1)) * lanes_direction

        to_lane = np.expand_dims(position, 1) - projection
        distance = np.linalg.norm(to_lane, axis=-1, keepdims=True)

        force = Config.U_cq * np.exp(-distance / Config.R_cq) * to_lane
        force = np.sum(force.reshape((self.num_bicycle, -1, 2)), axis=1)

        return force
    
    def calculate_leader_force_from_current_lane(self, index: int):
        speed = np.linalg.norm(self.velocity[index])

        desired_speed = self.desired_speed[index]
        com_decel = self.com_decel[index]
        max_accel = self.max_accel[index]
        space_headway = self.space_headway[index]

        leader_user = self.manager.get_leader(self.users[index], self.lane_index[index])

        accel = 0
        if leader_user is not None:
            follower_position, follower_speed = self.positions[index], speed
            leader_position, leader_speed = self.manager.get_road_user_info(leader_user)

            to_leader_distance = np.linalg.norm(follower_position - leader_position)
            delta_speed = follower_speed - leader_speed

            accel += -com_decel * (
                max((space_headway + follower_speed * Config.T_c +
                    ((follower_speed * delta_speed) / (2 * (max_accel * com_decel) ** 0.5))), 0) /
                max(to_leader_distance, space_headway)
            ) ** 2
        accel += max_accel * (1 - (speed / desired_speed) ** 4)

        force = accel * self.forward_vector_np[index]

        return force
    
    def calculate_repulsive_force_collision_avoidance(self):
        position = self.positions
        velocity = self.velocity

        diff_position = -1.0 * vector.each_diff(position)
        # diff_position -= (distance between bicycles)
        diff_velocity = vector.each_diff(velocity)

        to_bicycle_magnitude = np.linalg.norm(diff_position, axis=-1, keepdims=True) + 1e-8
        to_bicycle_direction = -1.0 * diff_position / to_bicycle_magnitude

        delta_speed = np.linalg.norm(diff_velocity, axis=-1, keepdims=True)

        B = 0.5 * ((to_bicycle_magnitude + np.linalg.norm(diff_position - diff_velocity * self.delta_time, axis=-1, keepdims=True)) ** 2 - \
                   (delta_speed * self.delta_time) ** 2) ** 0.5

        force = Config.U_kj * np.exp(-B / Config.R_kj) * to_bicycle_direction
        force = np.sum(force.reshape((self.num_bicycle, -1, 2)), axis=1)

        return force
    
    def calculate_overtaking_force(self, force_collision_avoidance: np.ndarray):
        force_collision_avoidance_magnitude = np.linalg.norm(force_collision_avoidance, axis=-1, keepdims=True) + 1e-8

        perpendicular_direction = force_collision_avoidance[:, [1, 0]] / force_collision_avoidance_magnitude
        perpendicular_direction[:, 0] *= -1

        force = Config.Alpha * force_collision_avoidance_magnitude * perpendicular_direction
        force = np.nan_to_num(force)

        return force
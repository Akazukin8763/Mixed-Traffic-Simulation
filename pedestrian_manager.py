from __future__ import annotations

import math
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


class PedestrianManager:
    FIXED_HEIGHT = 1.0

    class PedestrianData:
        def __init__(self, init_x, init_y, init_yaw, init_lane, target_x, target_y, forward_x, forward_y, right_x, right_y, crossing):
            self.init_x = init_x
            self.init_y = init_y
            self.init_yaw = init_yaw
            self.init_lane = init_lane
            self.target_x = target_x
            self.target_y = target_y
            self.forward_x = forward_x
            self.forward_y = forward_y
            self.right_x = right_x
            self.right_y = right_y
            self.crossing = crossing

    def __init__(self, traffic_manager: TrafficManager, world: World):
        self.manager: TrafficManager = traffic_manager

        self.world: World = world
        self.delta_time: float = self.world.delta_time

        self.pedestrians: List[carla.Actor] = []  # carla.Agent
        self.num_pedestrian: int = 0

        self.positions: np.ndarray = None  # np.ndarray: [[x, y], [x, y], ...]
        self.rotations: List[float] = None  # list: [yaw, yaw, ...]

        self.velocity: np.ndarray = None  # np.ndarray: [[x, y], [x, y], ...]
        self.desired_speed = None  # np.ndarray: [speed, speed, ...]

        self.forward_vector: List[carla.Vector3D] = None  # list: [carla.Vector3D, carla.Vector3D, ...]
        self.right_vector: List[carla.Vector3D] = None  # list: [carla.Vector3D, carla.Vector3D, ...]

        self.targets: np.ndarray = None  # np.ndarray: [[x, y], [x, y], ...]

        self.lane_index: List[int] = None  # list: [lane, lane, ...]

        self.users: List[UserID] = []  # list: [UserID, UserID, ...]

        # Data
        self.pedestrian_spawn_data: List[PedestrianManager.PedestrianData] = []  # PedestrianData
        self.__read_pedestrian_data()

    def __read_pedestrian_data(self):
        pedestrian_data_path = "configs/pedestrian.csv"
        pedestrian_data_rows = read_csv(pedestrian_data_path)

        for row in pedestrian_data_rows:
            self.pedestrian_spawn_data.append(
                PedestrianManager.PedestrianData(
                    init_x=float(row[0]),
                    init_y=float(row[1]),
                    init_yaw=float(row[2]),
                    init_lane=int(row[3]),
                    target_x=float(row[4]),
                    target_y=float(row[5]),
                    forward_x=float(row[6]),
                    forward_y=float(row[7]),
                    right_x=float(row[8]),
                    right_y=float(row[9]),
                    crossing=(row[10] == "True"),
                )
            )

    @property
    def vehicle_manager(self):
        return self.manager.vehicle_manager

    @property
    def bicycle_manager(self):
        return self.manager.bicycle_manager

    def spawn_agent(self) -> bool:
        blueprint = np.random.choice(self.world.blueprintsPedestrians)
        spawn_data = np.random.choice(self.pedestrian_spawn_data)

        spawn_position = carla.Location(spawn_data.init_x, spawn_data.init_y, PedestrianManager.FIXED_HEIGHT)
        spawn_rotation = carla.Rotation(0, spawn_data.init_yaw, 0)

        pedestrian = self.world.carla_world.try_spawn_actor(blueprint, carla.Transform(spawn_position, spawn_rotation))

        if pedestrian is not None:
            pedestrian.set_simulate_physics(False)
            self.pedestrians.append(pedestrian)

            if self.num_pedestrian == 0:
                self.positions = np.array([[spawn_data.init_x, spawn_data.init_y]])
                self.rotations = [spawn_data.init_yaw]

                self.velocity = np.array([[spawn_data.forward_x, spawn_data.forward_y]])
                self.desired_speed = np.array([np.random.normal(Config.Pedestrian_desired_speed_mean, Config.Pedestrian_desired_speed_std)])

                self.forward_vector = [carla.Vector3D(spawn_data.forward_x, spawn_data.forward_y, 0)]
                self.right_vector = [carla.Vector3D(spawn_data.right_x, spawn_data.right_y, 0)]

                self.targets = np.array([[spawn_data.target_x, spawn_data.target_y]])

                self.lane_index = [spawn_data.init_lane]

                self.crossings = np.array([spawn_data.crossing])
            else:
                self.positions = np.vstack((self.positions, [spawn_data.init_x, spawn_data.init_y]))
                self.rotations.append(spawn_data.init_yaw)

                self.velocity = np.vstack((self.velocity, [spawn_data.forward_x, spawn_data.forward_y]))
                self.desired_speed = np.hstack(
                    (self.desired_speed, np.random.normal(Config.Pedestrian_desired_speed_mean, Config.Pedestrian_desired_speed_std))
                )

                self.forward_vector.append(carla.Vector3D(spawn_data.forward_x, spawn_data.forward_y, 0))
                self.right_vector.append(carla.Vector3D(spawn_data.right_x, spawn_data.right_y, 0))

                self.targets = np.vstack((self.targets, [spawn_data.target_x, spawn_data.target_y]))

                self.lane_index.append(spawn_data.init_lane)

                self.crossings = np.hstack((self.crossings, spawn_data.crossing))

            user = UserID(RoadUser.Pedestrian, self.num_pedestrian)  # ID
            self.users.append(user)
            self.manager.lanes_road_users[spawn_data.init_lane].append(user)
            self.num_pedestrian += 1

            return True
        return False

    def step(self):
        if self.num_pedestrian == 0:
            return

        force = \
            self.calculate_driving_force() + \
            self.calculate_repulsive_force_with_distance() + \
            self.calculate_repulsive_force_with_velocity()
        crossing_force = \
            self.calculate_repulsive_force_from_vehicle(self.vehicle_manager.positions) + \
            self.calculate_repulsive_force_from_bicycle(self.bicycle_manager.positions)

        crossing_force[~self.crossings] = [0, 0]
        force += crossing_force

        # Update velocity
        self.velocity += force * self.delta_time
        self.constrain_velocity()

        # Update position
        self.positions += self.velocity * self.delta_time

        # Set the pedestrian transform in simulator
        self.update_transform()

    def update_transform(self):
        for idx, pedestrian in enumerate(self.pedestrians):
            if not pedestrian.is_alive:
                continue

            current_pos = carla.Location(pedestrian.get_location().x, pedestrian.get_location().y, PedestrianManager.FIXED_HEIGHT)
            future_pos = carla.Location(self.positions[idx][0], self.positions[idx][1], PedestrianManager.FIXED_HEIGHT)
            target_pos = carla.Location(self.targets[idx][0], self.targets[idx][1], PedestrianManager.FIXED_HEIGHT)

            to_forward = self.forward_vector[idx]
            to_future = normalize(future_pos - current_pos)
            pedestrian_direction = pedestrian.get_transform().get_forward_vector()

            current_lane_index = self.lane_index[idx]
            future_lane_index = self.manager.get_road_user_lane_index(future_pos)

            # Make the pedestrian direction change smoothly
            current_to_target_distance = current_pos.distance(target_pos)
            if current_to_target_distance < Config.Epsilon:
                # if the pedestrian is close to the target position, it needs to face the forward direction to wait to cross the lane
                # new_direction = pedestrian_direction + (to_forward - pedestrian_direction) * min(self.delta_time * 2.0, 1.0)

                self.positions[idx] = Config.Inf_position
                pedestrian.destroy()

                self.manager.refresh_lanes_road_users(self.users[idx], -1, to_add=False, to_remove=True)
                continue
            else:
                # otherwise, face the target position it wants to go
                new_direction = pedestrian_direction + (to_future - pedestrian_direction) * min(self.delta_time * 6.5, 1.0)

                if current_lane_index != future_lane_index:
                    self.lane_index[idx] = future_lane_index
                    self.manager.refresh_lanes_road_users(self.users[idx], future_lane_index, to_add=True, to_remove=True)
            new_yaw = math.degrees(math.atan2(new_direction.y, new_direction.x))

            # Set the pedestrian transform in simulator
            pedestrian.set_transform(carla.Transform(future_pos, carla.Rotation(0, new_yaw, 0)))

    def constrain_velocity(self):
        speeds = np.linalg.norm(self.velocity, axis=-1)
        factor = np.minimum(1.0, self.desired_speed / speeds)
        factor[speeds == 0] = 0.0
        return self.velocity * np.expand_dims(factor, -1)

    def calculate_driving_force(self):
        position = self.positions
        velocity = self.velocity
        target = self.targets

        direction, dist = vector.normalize(target - position)
        force = np.zeros((self.num_pedestrian, 2))

        force[dist > Config.Epsilon] = (direction * self.desired_speed.reshape((-1, 1)) - velocity.reshape((-1, 2)))[dist > Config.Epsilon, :]
        force[dist <= Config.Epsilon] = -1.0 * velocity[dist <= Config.Epsilon]
        force /= Config.Tau

        return force

    def calculate_repulsive_force_with_distance(self):
        # Circular specification
        position = self.positions

        # A and B are 1.0 respectively
        pos_diff = vector.each_diff(position)
        diff_direction, diff_length = vector.normalize(pos_diff)
        diff_direction *= np.exp(Config.Pedestrian_radius * 2 - (diff_length.reshape(-1, 1)) / 1.0)
        force = np.sum(diff_direction.reshape((self.num_pedestrian, -1, 2)), axis=1)

        return force

    def calculate_repulsive_force_with_velocity(self):
        # Ellipse specification
        position = self.positions
        velocity = self.velocity

        lambda_importance = 2.0
        gamma = 0.35
        n = 2
        n_prime = 3

        pos_diff = vector.each_diff(position)

        diff_direction, diff_length = vector.normalize(pos_diff)

        vel_diff = -1.0 * vector.each_diff(velocity)

        # compute interaction direction t_ij
        interaction_vec = lambda_importance * vel_diff + diff_direction
        interaction_direction, interaction_length = vector.normalize(interaction_vec)

        # compute angle theta (between interaction and position difference vector)
        theta = vector.vector_angles(interaction_direction) - vector.vector_angles(diff_direction)

        # compute model parameter B = gamma * ||D||
        B = gamma * interaction_length

        force_velocity_amount = np.exp(-1.0 * diff_length / B - np.square(n_prime * B * theta))
        force_angle_amount = -np.sign(theta) * np.exp(-1.0 * diff_length / B - np.square(n * B * theta))

        force_velocity = force_velocity_amount.reshape(-1, 1) * interaction_direction
        force_angle = force_angle_amount.reshape(-1, 1) * vector.left_normal(interaction_direction)

        force = force_velocity + force_angle
        force = np.sum(force.reshape((self.num_pedestrian, -1, 2)), axis=1)

        return force

    def calculate_repulsive_force_from_vehicle(self, vehicles_position: np.ndarray):
        position = self.positions

        diff_position = np.expand_dims(position, 1) - vehicles_position
        diff_position -= Config.Pedestrian_radius + 1.2

        to_pedestrian_magnitude = np.linalg.norm(diff_position, axis=-1, keepdims=True)
        to_pedestrian_direction = diff_position / to_pedestrian_magnitude

        # force = Config.Beta * (to_pedestrian_magnitude ** (-0.67)) * to_pedestrian_direction
        force = Config.Beta * np.exp(-to_pedestrian_magnitude * 0.20) * to_pedestrian_direction
        force = np.nan_to_num(force)
        force = np.sum(force.reshape((self.num_pedestrian, -1, 2)), axis=1)

        return force

    def calculate_repulsive_force_from_bicycle(self, bicycles_position: np.ndarray):
        position = self.positions

        diff_position = np.expand_dims(position, 1) - bicycles_position
        diff_position -= Config.Pedestrian_radius + 1.2

        to_pedestrian_magnitude = np.linalg.norm(diff_position, axis=-1, keepdims=True)
        to_pedestrian_direction = diff_position / to_pedestrian_magnitude

        # force = Config.Beta * (to_pedestrian_magnitude ** (-0.67)) * to_pedestrian_direction
        force = Config.Beta * np.exp(-to_pedestrian_magnitude * 0.20) * to_pedestrian_direction
        force = np.nan_to_num(force)
        force = np.sum(force.reshape((self.num_pedestrian, -1, 2)), axis=1)

        return force

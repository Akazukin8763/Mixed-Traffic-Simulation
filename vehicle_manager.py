from __future__ import annotations

from typing import TYPE_CHECKING, List

import carla
import numpy as np

from config import *
from region import Region
from road_user import RoadUser, UserID
from utils import *
from world import World

if TYPE_CHECKING:
    from traffic_manager import TrafficManager

class VehicleManager:
    FIXED_HEIGHT = 0.2

    class VehicleData:
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

        self.vehicles: List[carla.Actor] = []  # carla.Agent
        self.num_vehicle: int = 0

        self.positions: np.ndarray = None  # np.ndarray: [[x, y], [x, y], ...]

        self.speed = None  # np.ndarray: [speed, speed, ...]
        self.desired_speed = None  # np.ndarray: [speed, speed, ...]

        self.com_decel: List[float] = None  # comfortable deceleration, np.ndarray: [decel, decel, ...]
        self.max_accel: List[float] = None  # maximum acceleration, np.ndarray: [accel, accel, ...]
        self.space_headway: List[float] = None  # np.ndarray: [space, space, ...]

        self.forward_vector: List[carla.Vector3D] = None  # list: [carla.Vector3D, carla.Vector3D, ...]
        self.forward_vector_np: np.ndarray = None  # np.ndarray: [[x, y], [x, y], ...]

        self.targets: np.ndarray = None  # np.ndarray: [[x, y], [x, y], ...]

        self.lane_index: List[int] = None  # list: [lane, lane, ...]

        self.users: List[UserID] = []  # list: [UserID, UserID, ...]

        self.half_length: List[float] = None
        self.half_width: List[float] = None

        self.lane_changing_cooldown: List[int] = None  # list: [cooldown, cooldown, ...]
        self.lane_changing_target_lane_index: List[int] = None  # list: [lane, lane, ...]

        # Data
        self.vehicle_spawn_data: List[VehicleManager.VehicleData] = []  # VehicleData
        self.__read_vehicle_data()

    def __read_vehicle_data(self):
        vehicle_data_path = "configs/vehicle.csv"
        vehicle_data_rows = read_csv(vehicle_data_path)

        for row in vehicle_data_rows:
            self.vehicle_spawn_data.append(
                VehicleManager.VehicleData(
                    init_x=float(row[0]),
                    init_y=float(row[1]),
                    init_yaw=float(row[2]),
                    init_lane=int(row[3]),
                    target_x=float(row[4]),
                    target_y=float(row[5]),
                    forward_x=float(row[6]),
                    forward_y=float(row[7]),
                )
            )

    @property
    def pedestrian_manager(self):
        return self.manager.pedestrian_manager

    def spawn_agent(self) -> bool:
        blueprint = np.random.choice(self.world.blueprintsVehicles)
        spawn_data = np.random.choice(self.vehicle_spawn_data)

        spawn_position = carla.Location(spawn_data.init_x, spawn_data.init_y, VehicleManager.FIXED_HEIGHT)
        spawn_rotation = carla.Rotation(0, spawn_data.init_yaw, 0)

        vehicle = self.world.carla_world.try_spawn_actor(blueprint, carla.Transform(spawn_position, spawn_rotation))

        if vehicle is not None:
            vehicle.set_simulate_physics(False)
            self.vehicles.append(vehicle)

            if self.num_vehicle == 0:
                self.positions = np.array([[spawn_data.init_x, spawn_data.init_y]])

                self.velocity = np.array([[0.0, 0.0]])
                self.desired_speed = [np.random.normal(Config.Vehicle_desired_speed_mean, Config.Vehicle_desired_speed_std)]

                self.com_decel = [np.random.normal(Config.Vehicle_com_decel_mean, Config.Vehicle_com_decel_std)]
                self.max_accel = [np.random.normal(Config.Vehicle_min_accel_mean, Config.Vehicle_min_accel_std)]
                self.space_headway = [np.random.normal(Config.Vehicle_space_headway_mean, Config.Vehicle_space_headway_std)]

                self.forward_vector = [carla.Vector3D(spawn_data.forward_x, spawn_data.forward_y, 0)]
                self.forward_vector_np = np.array([[spawn_data.forward_x, spawn_data.forward_y]])

                self.targets = np.array([[spawn_data.target_x, spawn_data.target_y]])

                self.lane_index = [spawn_data.init_lane]

                half_length, half_width = get_car_size(vehicle)
                self.half_length = [half_length]
                self.half_width = [half_width]

                self.lane_changing_cooldown = [Config.Vehicle_lane_changing_cooldown]
                self.lane_changing_target_lane_index = [None]
            else:
                self.positions = np.vstack((self.positions, [spawn_data.init_x, spawn_data.init_y]))

                self.velocity = np.vstack((self.velocity, [0.0, 0.0]))
                self.desired_speed.append(np.random.normal(Config.Vehicle_desired_speed_mean, Config.Vehicle_desired_speed_std))

                self.com_decel.append(np.random.normal(Config.Vehicle_com_decel_mean, Config.Vehicle_com_decel_std))
                self.max_accel.append(np.random.normal(Config.Vehicle_min_accel_mean, Config.Vehicle_min_accel_std))
                self.space_headway.append(np.random.normal(Config.Vehicle_space_headway_mean, Config.Vehicle_space_headway_std))

                self.forward_vector.append(carla.Vector3D(spawn_data.forward_x, spawn_data.forward_y, 0))
                self.forward_vector_np = np.vstack((self.forward_vector_np, [spawn_data.forward_x, spawn_data.forward_y]))

                self.targets = np.vstack((self.targets, [spawn_data.target_x, spawn_data.target_y]))

                self.lane_index.append(spawn_data.init_lane)

                half_length, half_width = get_car_size(vehicle)
                self.half_length.append(half_length)
                self.half_width.append(half_width)

                self.lane_changing_cooldown.append(Config.Vehicle_lane_changing_cooldown)
                self.lane_changing_target_lane_index.append(None)

            user = UserID(RoadUser.Vehicle, self.num_vehicle)  # ID
            self.users.append(user)
            self.manager.lanes_road_users[spawn_data.init_lane].append(user)
            self.num_vehicle += 1

            return True
        return False

    def step(self):
        if self.num_vehicle == 0:
            return

        for idx, vehicle in enumerate(self.vehicles):
            if not vehicle.is_alive:
                continue

            force = (
                self.calculate_leader_force_from_current_lane(idx)
                + self.calculate_leader_force_from_adjacent_lane(idx)
                + self.calculate_lane_changing_force(idx)
            )

            # Update velocity
            self.velocity[idx] += force * self.delta_time

        # Calculate other force
        force = self.calculate_repulsive_force_from_lane(self.manager.lanes_line)

        # Update velocity
        self.velocity += force * self.delta_time
        self.constrain_velocity()

        # Update position
        self.positions += self.velocity * self.delta_time

        # Set the vehicle transform in simulator
        self.update_transform()

    def update_transform(self):
        for idx, vehicle in enumerate(self.vehicles):
            if not vehicle.is_alive:
                continue

            current_pos = carla.Location(vehicle.get_location().x, vehicle.get_location().y, VehicleManager.FIXED_HEIGHT)
            future_pos = carla.Location(self.positions[idx][0], self.positions[idx][1], VehicleManager.FIXED_HEIGHT)
            target_pos = carla.Location(self.targets[idx][0], self.targets[idx][1], VehicleManager.FIXED_HEIGHT)

            # current_to_target_distance = current_pos.distance(target_pos)
            # if current_to_target_distance < Config.Epsilon * 5:
            if current_pos.x < -40 or 40 < current_pos.x:
                self.positions[idx] = Config.Inf_position
                vehicle.destroy()

                # Remove the vehicle from the lane
                user = self.users[idx]
                for road_users in self.manager.lanes_road_users:
                    if user in road_users:
                        road_users.remove(user)

                continue

            # Set the vehicle transform in simulator
            vehicle.set_location(future_pos)
            # vehicle.set_transform(carla.Transform(position, carla.Rotation(0, self.init_yaw[idx], 0)))

    def constrain_velocity(self):
        # Limit the forward velocity in [0, desired_speed]
        forward_direction = self.forward_vector_np
        forward_speed = np.diagonal(np.dot(self.velocity, forward_direction.T), axis1=0, axis2=1)
        forward_velocity = forward_direction * forward_speed[:, np.newaxis]
        right_velocity = self.velocity - forward_velocity
        right_speed = np.linalg.norm(right_velocity, axis=-1)
        right_direction = right_velocity / right_speed[:, np.newaxis]

        forward_speed = np.clip(forward_speed, np.zeros(self.num_vehicle), self.desired_speed)
        right_speed[forward_speed < Config.Minimum_forward_speed_for_side_movement] = 0

        self.velocity = forward_speed[:, np.newaxis] * forward_direction + right_speed[:, np.newaxis] * right_direction

    def calculate_repulsive_force_from_lane(self, lanes_line: np.ndarray):
        position = self.positions

        lanes_direction = lanes_line[:, 0] - lanes_line[:, 1]
        to_lanes_start_point = np.expand_dims(position, 1) - lanes_line[:, 0]

        term1 = np.diagonal(np.dot(to_lanes_start_point, lanes_direction.T), axis1=1, axis2=2)
        term2 = np.diagonal(np.dot(lanes_direction, lanes_direction.T), axis1=0, axis2=1)
        projection = lanes_line[:, 0] + np.reshape(term1 / term2, (self.num_vehicle, -1, 1)) * lanes_direction

        to_lane = np.expand_dims(position, 1) - projection
        distance = np.linalg.norm(to_lane, axis=-1, keepdims=True)

        force = Config.U_cq * np.exp(-distance / Config.R_cq) * to_lane
        force = np.sum(force.reshape((self.num_vehicle, -1, 2)), axis=1)

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

    def calculate_leader_force_from_adjacent_lane(self, index: int):
        force = 0

        position = self.positions[index]

        current_lane_index = self.lane_index[index]
        adjacent_lane_index = [current_lane_index - 1, current_lane_index + 1]

        for lane_index in adjacent_lane_index:
            # if lane_index == 0 or lane_index == self.manager.NUM_LANE:
            #     continue

            self.manager.lanes_road_users[lane_index].append(self.users[index])
            self.manager.sort_lane_road_users(lane_index)
            leader_users = self.manager.get_leaders(self.users[index], lane_index)
            self.manager.lanes_road_users[lane_index].remove(self.users[index])

            for leader_user in leader_users:
                leader_position = self.manager.get_road_user_position_np(leader_user)

                to_self_magnitude = np.linalg.norm(position - leader_position)
                to_self_direction = (position - leader_position) / to_self_magnitude

                force += (
                    Config.U_cq
                    * np.exp(-to_self_magnitude / Config.R_cq)
                    * to_self_direction
                    * (5 if leader_user.user_type == RoadUser.Pedestrian else 1)
                )

                # user_position1 = carla.Location(leader_position[0], leader_position[1], 1.0)
                # user_position2 = carla.Location(position[0], position[1], 1.0)
                # self.manager.debug.draw_line(user_position1, user_position2, thickness=0.08, color=carla.Color(255, 0, 0), life_time=1/29)

        return force

    def calculate_lane_changing_force(self, index: int):
        force = 0

        if self.lane_changing_cooldown[index] > 0:
            if self.lane_changing_cooldown[index] > Config.Vehicle_lane_changing_cooldown - 10:
                self.velocity[index][1] *= 0.85
            self.lane_changing_cooldown[index] -= 1
            return force

        position = self.positions[index]

        current_lane_index = self.lane_index[index]
        current_leader_user = self.manager.get_leader(self.users[index], current_lane_index)
        current_follower_user = self.manager.get_follower(self.users[index], current_lane_index)

        # Not start the lane changing yet, need to check whether the vehicle can do the process
        if self.lane_changing_target_lane_index[index] is None:
            # Check adjacent lane
            adjacent_lane_index = [current_lane_index - 1, current_lane_index + 1]
            for lane_index in adjacent_lane_index:
                if lane_index not in [2, 3, 6, 7]:
                    continue

                # Add current vehicle into adjacent lane to get the leader and follower
                self.manager.lanes_road_users[lane_index].append(self.users[index])
                self.manager.sort_lane_road_users(lane_index)
                adjacent_leader_user = self.manager.get_leader(self.users[index], lane_index)
                adjacent_follower_user = self.manager.get_follower(self.users[index], lane_index)
                self.manager.lanes_road_users[lane_index].remove(self.users[index])  # Reset the lane road user state

                # If adjacent leader exists, check whether it has enough safe distance for lane changing
                if adjacent_leader_user is not None:
                    current_front = self.positions[index] + self.forward_vector_np[index] * self.half_length[index]
                    leader_back = self.manager.get_road_user_position_np(adjacent_leader_user, back=True)

                    if (
                        np.linalg.norm(leader_back - current_front) < Config.Vehicle_lane_changing_safe_distance
                        or np.dot(leader_back - current_front, self.forward_vector_np[index]) < 0
                    ):
                        continue

                    # user_position1 = carla.Location(current_front[0], current_front[1], 1.0)
                    # user_position2 = carla.Location(leader_back[0], leader_back[1], 1.0)
                    # self.manager.debug.draw_line(user_position1, user_position2, thickness=0.1, color=carla.Color(255, 0, 0), life_time=1/29)
                    # self.manager.debug.draw_string(user_position1, text=f'{current_lane_index}', color=carla.Color(255, 255, 0), life_time=1/29)
                    # self.manager.debug.draw_string(user_position2, text=f'{lane_index}', color=carla.Color(255, 255, 0), life_time=1/29)

                # If adjacent follower exists, check whether it has enough safe distance for lane changing
                if adjacent_follower_user is not None:
                    current_back = self.positions[index] - self.forward_vector_np[index] * self.half_length[index]
                    follower_front = self.manager.get_road_user_position_np(adjacent_follower_user, front=True)

                    if (
                        np.linalg.norm(current_back - follower_front) < Config.Vehicle_lane_changing_safe_distance
                        or np.dot(current_back - follower_front, self.forward_vector_np[index]) < 0
                    ):
                        continue

                    # user_position1 = carla.Location(current_back[0], current_back[1], 1.0)
                    # user_position2 = carla.Location(follower_front[0], follower_front[1], 1.0)
                    # self.manager.debug.draw_line(user_position1, user_position2, thickness=0.1, color=carla.Color(255, 0, 0), life_time=1/29)
                    # self.manager.debug.draw_string(user_position1, text=f'{current_lane_index}', color=carla.Color(255, 255, 0), life_time=1/29)
                    # self.manager.debug.draw_string(user_position2, text=f'{lane_index}', color=carla.Color(255, 255, 0), life_time=1/29)

                # Check whether the vehicle can go faster if do the lane changing
                expected_delta_speed = self.desired_speed[index] - np.linalg.norm(self.velocity[index])
                current_follower_expected_delta_speed = self.desired_speed[index]
                adjacent_follower_expected_delta_speed = self.desired_speed[index]

                if current_follower_user is not None:
                    current_follower_speed, current_follower_desired_speed = self.manager.get_road_user_speed(current_follower_user)
                    current_follower_expected_delta_speed = current_follower_desired_speed - current_follower_speed
                if adjacent_follower_user is not None:
                    adjacent_follower_speed, adjacent_follower_desired_speed = self.manager.get_road_user_speed(adjacent_follower_user)
                    adjacent_follower_expected_delta_speed = adjacent_follower_desired_speed - adjacent_follower_speed

                if expected_delta_speed + Config.P * (current_follower_expected_delta_speed + adjacent_follower_expected_delta_speed) <= Config.A_th:
                    continue

                # Setup the target lane
                self.lane_changing_target_lane_index[index] = lane_index
                break

        # Lane changing
        if self.lane_changing_target_lane_index[index] is not None:
            interrupt_lane_changing = False

            # TODO: Need another way to store the changing vector
            lane_index = self.lane_changing_target_lane_index[index]
            if lane_index == 2 or lane_index == 6:
                lane_changing_vector = np.array([0, -1])
            elif lane_index == 3 or lane_index == 7:
                lane_changing_vector = np.array([0, 1])

            # If the vehicle has already reached the target lane, update its lane info
            if lane_index != self.lane_index[index]:
                current_position = self.vehicles[index].get_location()
                target_region = self.manager.regions[lane_index]

                if target_region.is_in(current_position):
                    # Update the lane of current vehicle
                    self.manager.lanes_road_users[current_lane_index].remove(self.users[index])
                    self.manager.lanes_road_users[lane_index].append(self.users[index])
                    self.manager.sort_lane_road_users(current_lane_index)
                    self.manager.sort_lane_road_users(lane_index)
                    self.lane_index[index] = lane_index

            # Calculate the distance to target lane
            target_lane_center_line = (self.manager.lanes_line[lane_index] + self.manager.lanes_line[lane_index + 1]) / 2
            lane_direction = target_lane_center_line[0] - target_lane_center_line[1]
            to_lane_start_point = position - target_lane_center_line[0]

            term1 = np.dot(to_lane_start_point, lane_direction.T)
            term2 = np.dot(lane_direction, lane_direction.T)
            projection = target_lane_center_line[0] + term1 / term2 * lane_direction

            to_lane = projection - position
            distance = np.linalg.norm(to_lane, axis=-1, keepdims=True)

            # Check whether the danger region has any existed agent
            # If so, interrupt the lane changing process
            current_base = self.vehicles[index].get_location()
            current_base.z = VehicleManager.FIXED_HEIGHT
            region_expand_vector = carla.Vector3D(int(lane_changing_vector[0]), int(lane_changing_vector[1]), 0)

            current_region_front = current_base + self.forward_vector[index] * (self.half_length[index] + 1)
            current_region_back = current_base - self.forward_vector[index] * (self.half_length[index] + 1)
            target_region_front = current_region_front + region_expand_vector * float(distance + 1.8) + self.forward_vector[index] * 1.5
            target_region_back = current_region_back + region_expand_vector * float(distance + 1.8) - self.forward_vector[index] * 1.5

            danger_region = Region(current_region_front, current_region_back, target_region_back, target_region_front)
            if Config.DRAW_DEBUG:
                danger_region.draw_region(self.manager.debug, height=2.0, color=carla.Color(255, 0, 0), life_time=1 / 29)

            for user in self.manager.lanes_road_users[lane_index]:
                if user.user_type == RoadUser.Vehicle and user.user_ID == index:
                    continue

                # We need to check both front and back point, make sure the agent is not in the danger region
                check_position_front = self.manager.get_road_user_position(user, front=True)
                check_position_back = self.manager.get_road_user_position(user, back=True)

                # Interrupt the lane changing process
                if danger_region.is_in(check_position_front) or danger_region.is_in(check_position_back):
                    interrupt_lane_changing = True
                    break

            # If the vehicle has already reached the target lane, interrupt the lane changing process
            if distance < Config.Epsilon_lane_changing or np.dot(lane_changing_vector, to_lane) < 0:
                interrupt_lane_changing = True

            # Calculate the lane changing force
            if interrupt_lane_changing:
                # Cooldown the lane changing process
                self.lane_changing_cooldown[index] = Config.Vehicle_lane_changing_cooldown
                self.lane_changing_target_lane_index[index] = None

                self.velocity[index][1] *= 0.85
            else:
                # force = Config.U_cq * np.exp(-distance / Config.R_cq) * to_lane
                force = 0.03 * Config.U_cq * np.exp(distance * 0.15) * to_lane

                if Config.DRAW_DEBUG:
                    user_position1 = carla.Location((position + to_lane)[0], (position + to_lane)[1], 1.0)
                    user_position2 = carla.Location(position[0], position[1], 1.0)
                    self.manager.debug.draw_line(user_position1, user_position2, thickness=0.1, color=carla.Color(255, 255, 0), life_time=1 / 29)

        return force

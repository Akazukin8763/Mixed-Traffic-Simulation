import carla
import numpy as np
from tqdm import tqdm

from bicycle_manager import BicycleManager
from config import *
from pedestrian_manager import PedestrianManager
from region import Region
from road_user import RoadUser, UserID
from utils import process_time
from vehicle_manager import VehicleManager
from world import World


class TrafficManager:
    NUM_LANE = 10  # 0, 9 are sidewalks, 4, 5 is mid_sidewalk

    def __init__(self, pedestrian_spawn_rate=5, vehicle_spawn_rate=5, bicycle_spawn_rate=5, no_render_mode=False):
        self.world = World(no_render_mode=no_render_mode)
        self.debug = self.world.carla_world.debug
        self.current_frame = 0

        # Pedestrian
        self.pedestrian_spawn_rate = pedestrian_spawn_rate
        self.pedestrian_manager = PedestrianManager(self, self.world)

        # Vehicle
        self.vehicle_spawn_rate = vehicle_spawn_rate
        self.vehicle_manager = VehicleManager(self, self.world)

        # Bicycle
        self.bicycle_spawn_rate = bicycle_spawn_rate
        self.bicycle_manager = BicycleManager(self, self.world)

        # Lanes
        self.lanes_road_users = [[] for _ in range(TrafficManager.NUM_LANE)]

        min_x, max_x = -50, 50
        self.lanes_line = np.array(
            (
                [[max_x, 2.5], [min_x, 2.5]],
                [[max_x, 8.5], [min_x, 8.5]],
                [[max_x, 11.3], [min_x, 11.3]],
                [[max_x, 14.8], [min_x, 14.8]],
                [[max_x, 18.5], [min_x, 18.5]],
                [[max_x, 20.75], [min_x, 20.75]],
                [[max_x, 23], [min_x, 23]],
                [[max_x, 26.5], [min_x, 26.5]],
                [[max_x, 30], [min_x, 30]],
                [[max_x, 33], [min_x, 33]],
                [[max_x, 39], [min_x, 39]],
            )
        )

        self.regions = []
        for i in range(self.lanes_line.shape[0] - 1):
            region = Region(
                carla.Location(max_x, self.lanes_line[i][0][1], 0.0),
                carla.Location(min_x, self.lanes_line[i][0][1], 0.0),
                carla.Location(min_x, self.lanes_line[i + 1][0][1], 0.0),
                carla.Location(max_x, self.lanes_line[i + 1][0][1], 0.0),
            )
            self.regions.append(region)

        offset = 10
        self.regions_end = [
            carla.Location(min_x - offset, 5.5, 0.0),
            carla.Location(min_x - offset, 9.9, 0.0),
            carla.Location(min_x - offset, 13.05, 0.0),
            carla.Location(min_x - offset, 16.65, 0.0),
            carla.Location(min_x - offset, 19.625, 0.0),
            carla.Location(max_x + offset, 21.875, 0.0),
            carla.Location(max_x + offset, 24.75, 0.0),
            carla.Location(max_x + offset, 28.25, 0.0),
            carla.Location(max_x + offset, 31.5, 0.0),
            carla.Location(max_x + offset, 36, 0.0),
        ]

    @process_time
    def simulate(self, simulate_frame: int):
        progress = tqdm(range(simulate_frame), desc=f"Force-based Traffic Simulation", total=simulate_frame)

        for frame in progress:
            self.current_frame = frame

            self.__generate_at_frame(frame)
            self.world.carla_world.tick()

            self.sort_lanes_road_users()

            if Config.DRAW_DEBUG:
                for region in self.regions:
                    region.draw_region(self.debug, height=0.2, color=carla.Color(255, 0, 255), life_time=10)
                self.draw_lanes_road_users(height=0.2, color=carla.Color(0, 255, 255), life_time=1 / 29)

            self.vehicle_manager.step()
            self.pedestrian_manager.step()
            self.bicycle_manager.step()

    def __generate_at_frame(self, frame: int):
        # Generate the pedestrian
        if frame % self.pedestrian_spawn_rate == 0:
            self.pedestrian_manager.spawn_agent()

        # Generate the vehicle
        if frame % self.vehicle_spawn_rate == 0:
            self.vehicle_manager.spawn_agent()

        # Generate the bicycle
        if frame % self.bicycle_spawn_rate == 0:
            self.bicycle_manager.spawn_agent()

    def get_road_user_lane_index(self, position: carla.Location):
        for lane_index, region in enumerate(self.regions):
            if region.is_in(position):
                return lane_index
        return None

    def get_road_user_position(self, user: UserID, front: bool = False, back: bool = False):
        user_type = user.user_type
        user_ID = user.user_ID

        if user_type == RoadUser.Pedestrian:
            position = carla.Location(self.pedestrian_manager.positions[user_ID][0], self.pedestrian_manager.positions[user_ID][1], 0.0)
            user_position = position
        elif user_type == RoadUser.Vehicle:
            position = carla.Location(self.vehicle_manager.positions[user_ID][0], self.vehicle_manager.positions[user_ID][1], 0.0)
            if front:
                position += self.vehicle_manager.forward_vector[user_ID] * self.vehicle_manager.half_length[user_ID]
            if back:
                position -= self.vehicle_manager.forward_vector[user_ID] * self.vehicle_manager.half_length[user_ID]
            user_position = position
        elif user_type == RoadUser.Bicycle:
            position = carla.Location(self.bicycle_manager.positions[user_ID][0], self.bicycle_manager.positions[user_ID][1], 0.0)
            if front:
                position += self.bicycle_manager.forward_vector[user_ID] * self.bicycle_manager.half_length[user_ID]
            if back:
                position -= self.bicycle_manager.forward_vector[user_ID] * self.bicycle_manager.half_length[user_ID]
            user_position = position

        return user_position

    def get_road_user_position_np(self, user: UserID, front: bool = False, back: bool = False):
        user_type = user.user_type
        user_ID = user.user_ID

        if user_type == RoadUser.Pedestrian:
            position = self.pedestrian_manager.positions[user_ID].copy()
            user_position = position
        elif user_type == RoadUser.Vehicle:
            position = self.vehicle_manager.positions[user_ID].copy()
            position += self.vehicle_manager.forward_vector_np[user_ID] * self.vehicle_manager.half_length[user_ID] if front else 0
            position -= self.vehicle_manager.forward_vector_np[user_ID] * self.vehicle_manager.half_length[user_ID] if back else 0
            user_position = position
        elif user_type == RoadUser.Bicycle:
            position = self.bicycle_manager.positions[user_ID].copy()
            position += self.bicycle_manager.forward_vector_np[user_ID] * self.bicycle_manager.half_length[user_ID] if front else 0
            position -= self.bicycle_manager.forward_vector_np[user_ID] * self.bicycle_manager.half_length[user_ID] if back else 0
            user_position = position

        return user_position

    def get_road_user_speed(self, user: UserID):
        user_type = user.user_type
        user_ID = user.user_ID

        if user_type == RoadUser.Pedestrian:
            speed = np.linalg.norm(self.pedestrian_manager.velocity[user_ID])
            desired_speed = self.pedestrian_manager.desired_speed[user_ID]
        elif user_type == RoadUser.Vehicle:
            speed = np.linalg.norm(self.vehicle_manager.velocity[user_ID])
            desired_speed = self.vehicle_manager.desired_speed[user_ID]
        elif user_type == RoadUser.Bicycle:
            speed = np.linalg.norm(self.bicycle_manager.velocity[user_ID])
            desired_speed = self.bicycle_manager.desired_speed[user_ID]

        return speed, desired_speed

    def get_road_user_info(self, user: UserID):
        user_type = user.user_type
        user_ID = user.user_ID

        if user_type == RoadUser.Pedestrian:
            position = self.pedestrian_manager.positions[user_ID]

            return position, 0
        elif user_type == RoadUser.Vehicle:
            forward_vector = self.vehicle_manager.forward_vector_np[user_ID]
            half_length = self.vehicle_manager.half_length[user_ID]

            position = self.vehicle_manager.positions[user_ID] - forward_vector * half_length

            speed = np.linalg.norm(self.vehicle_manager.velocity[user_ID])

            return position, speed
        elif user_type == RoadUser.Bicycle:
            forward_vector = self.bicycle_manager.forward_vector_np[user_ID]
            half_length = self.bicycle_manager.half_length[user_ID]

            position = self.bicycle_manager.positions[user_ID] - forward_vector * half_length

            speed = np.linalg.norm(self.bicycle_manager.velocity[user_ID])

            return position, speed

    def refresh_lanes_road_users(self, user: UserID, current_lane_index: int, to_add: bool = True, to_remove: bool = True):
        for lane_index, road_users in enumerate(self.lanes_road_users):
            if to_add and lane_index == current_lane_index:
                if user not in road_users:
                    road_users.append(user)
            elif to_remove:
                if user in road_users:
                    road_users.remove(user)

    def sort_lanes_road_users(self):
        sorted_lanes_road_users = []

        for region_end, road_users in zip(self.regions_end, self.lanes_road_users):
            distance_to_end = []

            for road_user in road_users:
                user_position = self.get_road_user_position(road_user)

                distance_to_end.append(region_end.distance(user_position))

            sorted_road_users_index = np.argsort(distance_to_end)
            # sorted_road_users = []
            # for sorted_index in sorted_road_users_index:
            #     sorted_road_users.append(road_users[sorted_index])
            sorted_road_users = np.array(road_users)[sorted_road_users_index].tolist()
            sorted_lanes_road_users.append(sorted_road_users)

        self.lanes_road_users = sorted_lanes_road_users

    def sort_lane_road_users(self, lane_index: int):
        region_end = self.regions_end[lane_index]
        road_users = self.lanes_road_users[lane_index]
        distance_to_end = []

        for road_user in road_users:
            user_position = self.get_road_user_position(road_user)

            distance_to_end.append(region_end.distance(user_position))

        sorted_road_users_index = np.argsort(distance_to_end)
        # sorted_road_users = []
        # for sorted_index in sorted_road_users_index:
        #     sorted_road_users.append(road_users[sorted_index])
        sorted_road_users = np.array(road_users)[sorted_road_users_index].tolist()

        self.lanes_road_users[lane_index] = sorted_road_users

    def draw_lanes_road_users(self, height=0.0, thickness=0.1, color=carla.Color(0, 255, 0), life_time=1 / 30):
        for road_users in self.lanes_road_users:
            for i in range(len(road_users) - 1):
                user_position1 = self.get_road_user_position(road_users[i])
                user_position1.z = height

                user_position2 = self.get_road_user_position(road_users[i + 1])
                user_position2.z = height

                self.debug.draw_line(user_position1, user_position2, thickness=thickness, color=color, life_time=life_time)

    def get_leader(self, user: UserID, lane_index: int):
        """
        Get the leader in front of current vehicle on certain lane
        """
        road_users = self.lanes_road_users[lane_index]

        if user in road_users:
            current_index = road_users.index(user)
            if current_index > 0:
                return road_users[current_index - 1]

        return None

    def get_leaders(self, user: UserID, lane_index: int):
        """
        Get the leaders in front of current vehicle on certain lane
        """
        road_users = self.lanes_road_users[lane_index]

        if user in road_users:
            current_index = road_users.index(user)
            if current_index > 0:
                return road_users[:current_index]

        return []

    def get_follower(self, user: UserID, lane_index: int):
        """
        Get the follower in back of current vehicle on certain lane
        """
        road_users = self.lanes_road_users[lane_index]

        if user in road_users:
            current_index = road_users.index(user)
            if current_index < len(road_users) - 1:
                return road_users[current_index + 1]

        return None

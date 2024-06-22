from config import *
from traffic_manager import TrafficManager


def main():
    manager = None

    try:
        manager = TrafficManager(pedestrian_spawn_rate=Config.SPAWN_RATE_PEDESTRIAN,
                                 vehicle_spawn_rate=Config.SPAWN_RATE_VEHICLE,
                                 bicycle_spawn_rate=Config.SPAWN_RATE_BICYCLE)

        # Simulate
        manager.simulate(Config.SIMULATE_FRAME)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up the vehicles and pedestrians
        if manager is not None:
            for pedestrian in manager.pedestrian_manager.pedestrians:
                # print(f"Destroy Pedestrian: {pedestrian}")
                if pedestrian.is_alive:
                    pedestrian.destroy()
            print(f"Destroy Pedestrians Done")

            for vehicle in manager.vehicle_manager.vehicles:
                # print(f"Destroy Vehicle: {vehicle}")
                if vehicle.is_alive:
                    vehicle.destroy()
            print(f"Destroy Vehicles Done")

            for bicycle in manager.bicycle_manager.bicycles:
                # print(f"Destroy Bicycle: {bicycle}")
                if bicycle.is_alive:
                    bicycle.destroy()
            print(f"Destroy Bicycles Done")

            manager.world.destroy()


if __name__ == '__main__':
    main()

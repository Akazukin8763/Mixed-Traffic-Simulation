import re

import carla


class World:
    def __init__(self, draw_mode=False, simulate_mode=True, no_render_mode=False):
        self.draw_mode = draw_mode
        self.simulate_mode = simulate_mode
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(10.0)

        self.carla_world = self.client.get_world()
        self.carla_world_map = self.carla_world.get_map()
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.delta_time = 1 / 30
        self.spectator = self.carla_world.get_spectator()

        self.traffic_manager.set_synchronous_mode(True)
        settings = self.carla_world.get_settings()
        settings.synchronous_mode = True
        settings.no_rendering_mode = no_render_mode
        settings.fixed_delta_seconds = self.delta_time
        self.carla_world.apply_settings(settings)
        self.carla_world.set_weather(World.get_weather("Clear Noon"))

        # Blueprints
        self.blueprintsBicycles = [x for x in self.carla_world.get_blueprint_library().filter("vehicle.diamondback.century") if int(x.get_attribute("number_of_wheels")) == 2]
        self.blueprintsVehicles = [x for x in self.carla_world.get_blueprint_library().filter("vehicle.tesla.model3") if int(x.get_attribute("number_of_wheels")) == 4]
        self.blueprintsPedestrians = self.carla_world.get_blueprint_library().filter("walker.pedestrian.*")

    @staticmethod
    def get_weather(weather: str):
        rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
        name = lambda x: " ".join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]

        for w in presets:
            if name(w) == weather:
                return getattr(carla.WeatherParameters, w)
        raise KeyError

    def destroy(self):
        settings = self.carla_world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.carla_world.apply_settings(settings)

        if not self.simulate_mode:
            for actor in self.sensor_actors:
                if actor.is_alive:
                    actor.destroy()

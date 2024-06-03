try:
    import glob
    import os
    import sys

    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    
    import carla
except:
    raise ImportError

from utils import dot


class Region:
    def __init__(self, a: carla.Location, b: carla.Location, c: carla.Location, d: carla.Location):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self._center = None

    @property
    def max_x(self):
        return max(self.a.x, self.b.x, self.c.x, self.d.x)

    @property
    def min_x(self):
        return min(self.a.x, self.b.x, self.c.x, self.d.x)

    @property
    def center(self):
        if self._center is None:
            center_ab = (self.a + self.b) / 2
            center_cd = (self.c + self.d) / 2
            self._center = (center_ab + center_cd) / 2
            self._center.z = 0.0
        return self._center

    def is_in(self, position: carla.Location):
        v1 = self.b - self.a
        v2 = self.c - self.a
        v3 = self.d - self.a
        v = position - self.a

        if dot(v, v1) < 0 or dot(v, v2) < 0 or dot(v, v3) < 0:
            return False

        v1 = self.a - self.b
        v2 = self.c - self.b
        v3 = self.d - self.b
        v = position - self.b

        if dot(v, v1) < 0 or dot(v, v2) < 0 or dot(v, v3) < 0:
            return False

        v1 = self.a - self.c
        v2 = self.b - self.c
        v3 = self.d - self.c
        v = position - self.c

        if dot(v, v1) < 0 or dot(v, v2) < 0 or dot(v, v3) < 0:
            return False

        v1 = self.a - self.d
        v2 = self.b - self.d
        v3 = self.c - self.d
        v = position - self.d

        if dot(v, v1) < 0 or dot(v, v2) < 0 or dot(v, v3) < 0:
            return False

        return True
    
    def draw_region(self, debug, height=0.0, thickness=0.1, color=carla.Color(0, 255, 0), life_time=1/30):
        p1 = carla.Location(self.a.x, self.a.y, height)
        p2 = carla.Location(self.b.x, self.b.y, height)
        p3 = carla.Location(self.c.x, self.c.y, height)
        p4 = carla.Location(self.d.x, self.d.y, height)
        debug.draw_line(p1, p2, thickness=thickness, color=color, life_time=life_time)
        debug.draw_line(p2, p3, thickness=thickness, color=color, life_time=life_time)
        debug.draw_line(p3, p4, thickness=thickness, color=color, life_time=life_time)
        debug.draw_line(p4, p1, thickness=thickness, color=color, life_time=life_time)

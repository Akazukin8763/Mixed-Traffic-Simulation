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

import csv
import time

import numpy as np


def process_time(old):
    def new(*args, **kwargs):
        before = time.time()

        result = old(*args, **kwargs)

        after = time.time()
        print(f"{old.__name__} processing time: {after - before} seconds")

        return result
    return new


def read_csv(filepath):
    result = []
    with open(filepath, newline='') as csvfile:
        rows = csv.reader(csvfile)
        header = next(rows)  # skip the headers
        for row in rows:
            result.append(row)
    return result


def dot(v1: carla.Vector3D, v2: carla.Vector3D):
	return	v1.x * v2.x + v1.y * v2.y + v1.z * v2.z	 


def normalize(vec) -> carla.Vector3D:
	length = (vec.x**2 + vec.y**2 + vec.z**2)**0.5
	
	if length != 0:
		vec /= length
	else:
		vec = carla.Vector3D(0, 0, 0)
          
	return vec

def get_matrix(transform):
	"""
	Creates matrix from carla transform.
	"""
	rotation = transform.rotation
	location = transform.location
	c_y = np.cos(np.radians(rotation.yaw))
	s_y = np.sin(np.radians(rotation.yaw))
	c_r = np.cos(np.radians(rotation.roll))
	s_r = np.sin(np.radians(rotation.roll))
	c_p = np.cos(np.radians(rotation.pitch))
	s_p = np.sin(np.radians(rotation.pitch))
	matrix = np.matrix(np.identity(4))
	matrix[0, 3] = location.x
	matrix[1, 3] = location.y
	matrix[2, 3] = location.z
	matrix[0, 0] = c_p * c_y
	matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
	matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
	matrix[1, 0] = s_y * c_p
	matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
	matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
	matrix[2, 0] = s_p
	matrix[2, 1] = -c_p * s_r
	matrix[2, 2] = c_p * c_r
	return matrix

def get_bounding_box_coords(vehicle):
	cords = np.zeros((8, 4))
	extent = vehicle.bounding_box.extent
	cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
	cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
	cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
	cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
	cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
	cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
	cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
	cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
	
	bb_transform = carla.Transform(vehicle.bounding_box.location)
	bb_vehicle_matrix = get_matrix(bb_transform)
	vehicle_world_matrix = get_matrix(vehicle.get_transform())
	bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
	cords_x_y_z = np.dot(bb_world_matrix, np.transpose(cords))[:3, :]
	
	return cords_x_y_z
      
def get_car_size(vehicle):
	cords_x_y_z = get_bounding_box_coords(vehicle)
	boxCoord_1 = cords_x_y_z[:, 0]
	boxCoord_2 = cords_x_y_z[:, 1]
	boxCoord_3 = cords_x_y_z[:, 2]
	boxCoord_4 = cords_x_y_z[:, 3]
	
	forward_vector = vehicle.get_transform().get_forward_vector()
	forward_vector.z = 0
	right_vector = vehicle.get_transform().get_right_vector()
	right_vector.z = 0
	
	pos = vehicle.get_location()
	pos.z = 0
	
	a = carla.Location(float(boxCoord_1[0]), float(boxCoord_1[1]), 0) 
	c = carla.Location(float(boxCoord_4[0]), float(boxCoord_4[1]), 0) 
		
	b = carla.Location(float(boxCoord_2[0]), float(boxCoord_2[1]), 0) 
	d = carla.Location(float(boxCoord_3[0]), float(boxCoord_3[1]), 0) 
	
	dis = pos - (a + b) / 2
	dis2 = pos - (b + d) / 2
	
	if (dis.x**2 + dis.y**2)**0.5 > (dis2.x**2 + dis2.y**2)**0.5:
		return (dis.x**2 + dis.y**2)**0.5,(dis2.x**2 + dis2.y**2)**0.5	
	else:
		return (dis2.x**2 + dis2.y**2)**0.5,(dis.x**2 + dis.y**2)**0.5
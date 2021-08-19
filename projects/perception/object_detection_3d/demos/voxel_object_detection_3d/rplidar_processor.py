from rplidar import RPLidar as RPLidarAPI
import numpy as np
import math
from opendr.engine.data import PointCloud
import threading


def create_point_cloud(scan, z=0):

    points = np.empty((len(scan), 4), dtype=np.float32)

    for i, s in enumerate(scan):
        r, angle_degrees, distance_mm = s

        angle_rad = angle_degrees * math.pi / 180

        y = math.sin(angle_rad) * distance_mm / 1000
        x = math.cos(angle_rad) * distance_mm / 1000

        points[i] = [x, y, z, r / 16]

    return points


class RPLidar():
    def __init__(self, port, baudrate=115200, timeout=1):
        
        lidar = RPLidarAPI(port=port, baudrate=baudrate, timeout=timeout)

        self.lidar = lidar

        lidar.clean_input()

        info = lidar.get_info()
        print(info)

        health = lidar.get_health()
        print(health)

        self.iterate_thread = threading.Thread(target=self.__itereate_scans)
        self.iterate_thread.start()
        self.lock = threading.Lock()

        self.last_point_cloud = np.zeros((0, 3), dtype=np.float32)

        self.running = True

    def __itereate_scans(self):

        for scan in self.lidar.iter_scans(min_len=1):
            
            pc = create_point_cloud(scan)

            with self.lock:
                self.last_point_cloud = pc

            if not self.running:
                return

    def next(self):

        with self.lock:
            return PointCloud(self.last_point_cloud)

    def stop(self):

        self.running = False
        self.iterate_thread.join()

        self.lidar.stop()
        self.lidar.stop_motor()
        self.lidar.disconnect()

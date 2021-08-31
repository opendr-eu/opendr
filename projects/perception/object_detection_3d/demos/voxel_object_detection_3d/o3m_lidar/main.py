from .o3m_lidar import O3MLidar
import socket


ip = "169.254.160.76"
port = 42000
buffer_size = 1460

def main():

    lidar = O3MLidar(ip, port, buffer_size, output_mode="point_cloud")

    i = 0

    while True:
        point_cloud = lidar.next()
        if len(point_cloud) > 0:
            print(i, lidar.next()[0])
            i += 1

main()
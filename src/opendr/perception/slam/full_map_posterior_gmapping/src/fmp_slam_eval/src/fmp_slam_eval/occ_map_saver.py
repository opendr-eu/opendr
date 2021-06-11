# ROS Libraries
import rospy

# ROS Messages
from nav_msgs.msg import OccupancyGrid

# Math Libraries
import matplotlib.pyplot as plt

from datetime import datetime
import os

from map_simulator.utils import map_msg_to_numpy, mkdir_p


class OccMapSaver:

    def __init__(self):

        rospy.init_node('occ_map_saver')

        path_prefix = rospy.get_param("~path_prefix", "map")
        self._file_prefix = rospy.get_param('~file_prefix', default="map")

        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        default_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        default_path = os.path.join(default_path, 'OccMap')
        default_path = os.path.join(default_path, timestamp + "_" + path_prefix)

        save_dir = rospy.get_param("~save_dir", default_path)
        save_dir = os.path.expanduser(save_dir)
        save_dir = os.path.expandvars(save_dir)
        save_dir = os.path.normpath(save_dir)

        if not os.path.exists(save_dir):
            mkdir_p(save_dir)

        self._save_dir = save_dir

        rospy.Subscriber("map", OccupancyGrid, self._save_map, queue_size=1)

        rospy.spin()

    def _save_map(self, msg):

        occ_map = map_msg_to_numpy(msg)

        occ_map[occ_map == -1] = 50

        file_name = "{}_s{}.png".format(self._file_prefix, msg.header.seq)
        file_path = os.path.join(self._save_dir, file_name)

        plt.imsave(file_path, occ_map, vmin=0, vmax=100, cmap='binary')

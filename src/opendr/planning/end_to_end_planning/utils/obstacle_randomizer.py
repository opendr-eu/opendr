# Copyright 2020-2023 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R
import webots_ros.srv
from geometry_msgs.msg import Point, Quaternion


class ObstacleRandomizer():
    def __init__(self, model_name):
        self.number_of_obstacles = 10
        self.model_name = model_name
        self.ros_srv_get_from_def = rospy.ServiceProxy("/supervisor/get_from_def",
                                                       webots_ros.srv.supervisor_get_from_def)
        self.ros_srv_get_field = rospy.ServiceProxy("/supervisor/node/get_field",
                                                    webots_ros.srv.node_get_field)
        self.ros_srv_field_set_v3 = rospy.ServiceProxy("/supervisor/field/set_vec3f",
                                                       webots_ros.srv.field_set_vec3f)
        self.ros_srv_field_set_rotation = rospy.ServiceProxy("/supervisor/field/set_rotation",
                                                             webots_ros.srv.field_set_rotation)
        self.ros_srv_field_set_float = rospy.ServiceProxy("/supervisor/field/set_float",
                                                          webots_ros.srv.field_set_float)
        try:
            self.cyl_solid_nodes = [self.ros_srv_get_from_def(name="cyl" + str(i)).node for i in range(1, 6)]
            self.cyl_geometry_nodes = [self.ros_srv_get_from_def(name="cyl_geo" + str(i)).node for i in range(1, 6)]
            self.box_solid_nodes = [self.ros_srv_get_from_def(name="box" + str(i)).node for i in range(1, 6)]
            self.box_geometry_nodes = [self.ros_srv_get_from_def(name="box_geo" + str(i)).node for i in range(1, 6)]
            self.sph_solid_nodes = [self.ros_srv_get_from_def(name="sph" + str(i)).node for i in range(1, 6)]
            self.sph_geometry_nodes = [self.ros_srv_get_from_def(name="sph_geo" + str(i)).node for i in range(1, 6)]
            self.wall_nodes = [self.ros_srv_get_from_def(name='wall' + str(i)).node for i in range(1, 3)]
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        try:
            self.cyl_solid_translation_fields = [self.ros_srv_get_field(node, 'translation', False).field for node in
                                                 self.cyl_solid_nodes]
            self.cyl_solid_rotation_fields = [self.ros_srv_get_field(node, 'rotation', False).field for node in
                                              self.cyl_solid_nodes]
            self.cyl_geometry_radius_fields = [self.ros_srv_get_field(node, 'radius', False).field for node in
                                               self.cyl_geometry_nodes]
            self.cyl_geometry_height_fields = [self.ros_srv_get_field(node, 'height', False).field for node in
                                               self.cyl_geometry_nodes]
            self.box_solid_translation_fields = [self.ros_srv_get_field(node, 'translation', False).field for node in
                                                 self.box_solid_nodes]
            self.box_solid_rotation_fields = [self.ros_srv_get_field(node, 'rotation', False).field for node in
                                              self.box_solid_nodes]
            self.box_geometry_size_fields = [self.ros_srv_get_field(node, 'size', False).field for node in
                                             self.box_geometry_nodes]
            self.sph_solid_translation_fields = [self.ros_srv_get_field(node, 'translation', False).field for node in
                                                 self.sph_solid_nodes]
            self.sph_geometry_radius_fields = [self.ros_srv_get_field(node, 'radius', False).field for node in
                                               self.sph_geometry_nodes]
            self.wall_translation_fields = [self.ros_srv_get_field(node, 'translation', False).field for node in
                                            self.wall_nodes]
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        self.keep_configuration = {}

    def randomize_environment(self, with_walls=None, number_of_obstacles=None, save_config_dir=None):
        if with_walls is None:
            with_walls = np.random.choice([True, False])
        self.randomize_walls(with_walls)
        if number_of_obstacles is None:
            number_of_obstacles = np.random.randint(2, 7)
        self.keep_configuration['with_walls'] = with_walls
        self.keep_configuration['number_of_obstacles'] = number_of_obstacles
        c = np.random.choice(3, number_of_obstacles)
        number_of_cylinders = np.sum(c == 0)
        number_of_boxs = np.sum(c == 1)
        number_of_spheres = np.sum(c == 2)
        while number_of_spheres > 5 or number_of_boxs > 5 or number_of_cylinders > 5:
            c = np.random.choice(3, number_of_obstacles)
            number_of_cylinders = np.sum(c == 0)
            number_of_boxs = np.sum(c == 1)
            number_of_spheres = np.sum(c == 2)
        self.keep_configuration['number_of_cylinders'] = number_of_cylinders
        self.keep_configuration['number_of_boxs'] = number_of_boxs
        self.keep_configuration['number_of_spheres'] = number_of_spheres
        self.randomize_cylinders(number_of_cylinders)
        self.randomize_boxs(number_of_boxs)
        self.randomize_spheres(number_of_spheres)
        if save_config_dir is not None:
            np.save(save_config_dir, self.keep_configuration)

    def randomize_cylinders(self, num=5, lower_size=1, higher_size=3):
        for i in range(num):
            t_field = self.cyl_solid_translation_fields[i]
            p = Point()
            p.y = np.random.normal(0, 2.5)
            p.x = -np.random.uniform(-7, 20)
            p.z = np.random.uniform(2, 3)
            self.ros_srv_field_set_v3(t_field, 0, p)
            rot_field = self.cyl_solid_rotation_fields[i]
            q = Quaternion()
            [q.x, q.y, q.z, q.w] = R.random().as_quat()
            self.ros_srv_field_set_rotation(rot_field, 0, q)
            rad_field = self.cyl_geometry_radius_fields[i]
            rad = np.random.uniform(lower_size / 2, higher_size / 2)
            self.ros_srv_field_set_float(rad_field, 0, rad)
            h_field = self.cyl_geometry_height_fields[i]
            h = np.random.uniform(lower_size, higher_size)
            self.ros_srv_field_set_float(h_field, 0, h)
            self.keep_configuration["cyl" + str(i) + "p"] = p
            self.keep_configuration["cyl" + str(i) + "rot"] = q
            self.keep_configuration["cyl" + str(i) + "rad"] = rad
            self.keep_configuration["cyl" + str(i) + "h"] = h
        for i in range(num, 5):
            t_field = self.cyl_solid_translation_fields[i]
            p = Point()
            p.z = -10
            self.ros_srv_field_set_v3(t_field, 0, p)

    def randomize_boxs(self, num=5, lower_size=0.5, higher_size=2.5):
        for i in range(num):
            t_field = self.box_solid_translation_fields[i]
            p = Point()
            p.y = np.random.normal(0, 2.5)
            p.x = -np.random.uniform(-7, 20)
            p.z = np.random.uniform(2, 3)
            self.ros_srv_field_set_v3(t_field, 0, p)
            rot_field = self.box_solid_rotation_fields[i]
            q = Quaternion()
            [q.x, q.y, q.z, q.w] = R.random().as_quat()
            self.ros_srv_field_set_rotation(rot_field, 0, q)
            size_field = self.box_geometry_size_fields[i]
            size = Point()
            size.x = np.random.uniform(lower_size, higher_size)
            size.z = np.random.uniform(lower_size, higher_size)
            size.y = np.random.uniform(lower_size, higher_size)
            self.ros_srv_field_set_v3(size_field, 0, size)
            self.keep_configuration["box" + str(i) + "p"] = p
            self.keep_configuration["box" + str(i) + "rot"] = q
            self.keep_configuration["box" + str(i) + "size"] = size
        for i in range(num, 5):
            t_field = self.box_solid_translation_fields[i]
            p = Point()
            p.z = -10
            self.ros_srv_field_set_v3(t_field, 0, p)

    def randomize_spheres(self, num=5, lower_radius=0.5, higher_radius=1.5):
        for i in range(num):
            t_field = self.sph_solid_translation_fields[i]
            p = Point()
            p.y = np.random.normal(0, 2.5)
            p.x = -np.random.uniform(-7, 20)
            p.z = np.random.uniform(2, 3)
            self.ros_srv_field_set_v3(t_field, 0, p)
            rad_field = self.sph_geometry_radius_fields[i]
            rad = np.random.uniform(lower_radius, higher_radius)
            self.ros_srv_field_set_float(rad_field, 0, rad)
            self.keep_configuration["sphere" + str(i) + "p"] = p
            self.keep_configuration["sphere" + str(i) + "rad"] = rad
        for i in range(num, 5):
            t_field = self.sph_solid_translation_fields[i]
            p = Point()
            p.z = -10
            self.ros_srv_field_set_v3(t_field, 0, p)

    def randomize_walls(self, with_walls=True, lower_width=4, higher_width=10):
        field = self.wall_translation_fields[0]
        p = Point()
        width = np.random.uniform(lower_width, higher_width)
        self.keep_configuration["wall_width"] = width
        if with_walls:
            p.z = -1
        else:
            p.z = -9
        p.y = -width / 2
        p.x = -4
        self.ros_srv_field_set_v3(field, 0, p)
        p.y = width / 2
        field = self.wall_translation_fields[1]
        self.ros_srv_field_set_v3(field, 0, p)

    def reload_environment(self, load_dir):
        conf = np.load(load_dir, allow_pickle=True).item()
        self.randomize_walls(with_walls=conf["with_walls"], lower_width=conf["wall_width"],
                             higher_width=conf["wall_width"])
        # set cylinders
        for i in range(conf["number_of_cylinders"]):
            t_field = self.cyl_solid_translation_fields[i]
            p = conf["cyl" + str(i) + "p"]
            self.ros_srv_field_set_v3(t_field, 0, p)
            rot_field = self.cyl_solid_rotation_fields[i]
            q = conf["cyl" + str(i) + "rot"]
            self.ros_srv_field_set_rotation(rot_field, 0, q)
            rad_field = self.cyl_geometry_radius_fields[i]
            rad = conf["cyl" + str(i) + "rad"]
            self.ros_srv_field_set_float(rad_field, 0, rad)
            h_field = self.cyl_geometry_height_fields[i]
            h = conf["cyl" + str(i) + "h"]
            self.ros_srv_field_set_float(h_field, 0, h)
        for i in range(conf["number_of_cylinders"], 5):
            t_field = self.cyl_solid_translation_fields[i]
            p = Point()
            p.y = -10
            self.ros_srv_field_set_v3(t_field, 0, p)
        # set boxes
        for i in range(conf["number_of_boxs"]):
            t_field = self.box_solid_translation_fields[i]
            p = conf["box" + str(i) + "p"]
            self.ros_srv_field_set_v3(t_field, 0, p)
            rot_field = self.box_solid_rotation_fields[i]
            q = conf["box" + str(i) + "rot"]
            self.ros_srv_field_set_rotation(rot_field, 0, q)
            size_field = self.box_geometry_size_fields[i]
            size = conf["box" + str(i) + "size"]
            self.ros_srv_field_set_v3(size_field, 0, size)
        for i in range(conf["number_of_boxs"], 5):
            t_field = self.box_solid_translation_fields[i]
            p = Point()
            p.y = -10
            self.ros_srv_field_set_v3(t_field, 0, p)
        # set spheres
        for i in range(conf["number_of_spheres"]):
            t_field = self.sph_solid_translation_fields[i]
            p = conf["sphere" + str(i) + "p"]
            self.ros_srv_field_set_v3(t_field, 0, p)
            rad_field = self.sph_geometry_radius_fields[i]
            rad = conf["sphere" + str(i) + "rad"]
            self.ros_srv_field_set_float(rad_field, 0, rad)
        for i in range(conf["number_of_spheres"], 5):
            t_field = self.sph_solid_translation_fields[i]
            p = Point()
            p.y = -10
            self.ros_srv_field_set_v3(t_field, 0, p)

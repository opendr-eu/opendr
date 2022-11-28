import rclpy


class MyRobotDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot

        self.__gps = self.__robot.getDevice('gps1')
        self.__imu = self.__robot.getDevice('inertial_unit')

        rclpy.init(args=None)
        self.__node = rclpy.create_node('my_robot_driver')

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)

        roll, pitch, yaw = self.__imu.getRollPitchYaw()
        v1, v2, v3 = self.__gps.getValues()

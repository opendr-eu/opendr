"""robotti sample controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Motor, Camera, Lidar, LidarPoint, GPS, CameraRecognitionObject

MAX_SPEED = 6.28

# create the instances.

robot = Robot()
cameraTop = Camera("front_top_camera")
cameraBottom = Camera("front_bottom_camera")
cameraRear = Camera("rear_camera")
cameraDownwards1 = Camera("camera_downwards1")
cameraDownwards2 = Camera("camera_downwards2")
cameraDownwards3 = Camera("camera_downwards3")
cameraDownwards4 = Camera("camera_downwards4")
lidar = Lidar("velodyne")
gps = robot.getDevice("Hemisphere_v500")

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

leftMotorFront = robot.getDevice("left_front_wheel_joint_motor")
leftMotorRear = robot.getDevice("left_rear_wheel_joint_motor")
rightMotorFront = robot.getDevice("right_front_wheel_joint_motor")
rightMotorRear = robot.getDevice("right_rear_wheel_joint_motor")

# enable all the devices
cameraTop.enable(4 * timestep)
cameraBottom.enable(10 * timestep)
# cameraBottom.recognitionEnable(10 * timestep)
cameraRear.enable(10 * timestep)
robot.step(2 * timestep) 
cameraDownwards1.enable(10 * timestep)
robot.step(2 * timestep) 
cameraDownwards2.enable(10 * timestep)
robot.step(2 * timestep) 
cameraDownwards3.enable(10 * timestep)
robot.step(2 * timestep) 
cameraDownwards4.enable(10 * timestep)
lidar.enable(timestep)
lidar.enablePointCloud()
gps.enable(timestep)

# set motor veloctiy control
leftMotorFront.setPosition(float('inf'))
leftMotorRear.setPosition(float('inf'))
rightMotorFront.setPosition(float('inf'))
rightMotorRear.setPosition(float('inf'))
leftMotorFront.setVelocity(0.1 * MAX_SPEED)
leftMotorRear.setVelocity(0.1 * MAX_SPEED)
rightMotorFront.setVelocity(0.1 * MAX_SPEED)
rightMotorRear.setVelocity(0.1 * MAX_SPEED)

while robot.step(timestep) != -1:
    pass

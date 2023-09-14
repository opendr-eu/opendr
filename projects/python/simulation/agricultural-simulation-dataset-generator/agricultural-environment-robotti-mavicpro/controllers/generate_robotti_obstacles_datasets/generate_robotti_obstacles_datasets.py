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

from controller import Supervisor, Camera, Lidar, GPS

useRobotti = True

if (useRobotti):

    import math
    import random
    import pathlib
    import os
    import json
    from datetime import date

    import time

    obj = time.gmtime(0)
    epoch = time.asctime(obj)
    # print("The epoch is:",epoch)

    FIELD_SIZE = [40, 14]
    DATASET_NAME = 'dataset_location/UGV'

    MAX_RECORDS_PER_SCENARIO = 19300
    OBSTACLES_PER_SCENARIO = 12
    STOP_ON = 193

    # set lighting conditions
    backgrounds = [
        'noon_cloudy_countryside']  # options: noon_cloudy_countryside, dawn_cloudy_empty, noon_stormy_empty, dusk

    # enable_fog = bool(random.getrandbits(1))
    enable_fog = False  # options: False, True

    global already_set_the_velocity
    already_set_the_velocity = False

    def save_datasets_info(count, categories):
        print(count)
        with open(os.path.join(DATASET_NAME, "data.json"), 'w') as f:
            data = {
                'version': '1.0',
                'source': 'Webots R2023a',
                'updated': str(date.today()),
                'categories': categories,
                'size': index - 1
            }
            f.write(json.dumps(data))

    def save_device_measurements_old(index, cameras, lidar, gps, objects, static_objects):
        index = index.zfill(6)
        # RGB Camera images
        for device in cameras:
            dir_name = os.path.join(DATASET_NAME, device.getName())
            pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
            device.saveImage(os.path.join(dir_name, index + ".jpg"), 100)

            dir_name = os.path.join(DATASET_NAME, 'annotations', device.getName())
            pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
            device.saveRecognitionSegmentationImage(os.path.join(dir_name, index + "_segmented.jpg"), 100)

            with open(os.path.join(dir_name, index + "_annotations.txt"), 'w') as f:
                f.write('# YOLO annotations format: <object-class> <x> <y> <width> <height>\n')
                for object in device.getRecognitionObjects():
                    position = object.getPositionOnImage()
                    size = object.getSizeOnImage()
                    f.write('"{}" {} {} {} {}\n'.format(object.getModel(), position[0], position[1], size[0], size[1]))

        # Lidar Point Cloud
        dir_name = os.path.join(DATASET_NAME, lidar.getName())
        pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dir_name, index + ".pcd"), 'w') as f:
            f.write('VERSION .7\n')
            f.write('FIELDS x y z\n')
            f.write('SIZE 4 4 4\n')
            f.write('TYPE F F F\n')
            f.write('COUNT 1 1 1\n')
            f.write(f'WIDTH {lidar.getHorizontalResolution()}\n')
            f.write(f'HEIGHT {lidar.getNumberOfLayers()}\n')
            f.write(f'POINTS {lidar.getNumberOfPoints()}\n')
            f.write('DATA ascii\n')
            points = lidar.getPointCloud()
            for point in points:
                f.write(f'{point.x} {point.y} {point.z}\n')

        # GPS
        dir_name = os.path.join(DATASET_NAME, gps.getName())
        pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dir_name, index + ".txt"), 'w') as f:
            value = gps.getValues()
            f.write(f'{value[0]} {value[1]} {value[2]}')

        # Objects position
        dir_name = os.path.join(DATASET_NAME, 'annotations', 'scene')
        pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dir_name, index + ".txt"), 'w') as f:
            for object in objects:
                position = object.getPosition()
                f.write(f'"{object.getTypeName()}" {position[0]} {position[1]} {position[2]}\n')
            for object in static_objects:
                position = object['position']
                type_name = object['type name']
                f.write(f'"{type_name}" {position[0]} {position[1]} {position[2]}\n')

    def save_device_measurements(index, second, cameras, lidar, gps, objects, static_objects):
        index = index.zfill(6)

        number = int(index)

        index = str(second) + '_' + str(index)

        if (number % 3 == 0):

            # RGB Camera images
            for device in cameras:
                dir_name = os.path.join(DATASET_NAME, device.getName())
                pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
                device.saveImage(os.path.join(dir_name, index + ".jpg"), 100)

                dir_name = os.path.join(DATASET_NAME, 'annotations', device.getName())
                pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
                device.saveRecognitionSegmentationImage(os.path.join(dir_name, index + "_segmented.jpg"), 100)

                with open(os.path.join(dir_name, index + "_annotations.txt"), 'w') as f:
                    f.write('# YOLO annotations format: <object-class> <x> <y> <width> <height>\n')
                    for object in device.getRecognitionObjects():
                        position = object.getPositionOnImage()
                        size = object.getSizeOnImage()
                        f.write(
                            '"{}" {} {} {} {}\n'.format(object.getModel(), position[0], position[1], size[0], size[1]))


        if (number % 5 == 0):

            # Lidar Point Cloud
            dir_name = os.path.join(DATASET_NAME, lidar.getName())
            pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(dir_name, index + ".pcd"), 'w') as f:
                f.write('VERSION .7\n')
                f.write('FIELDS x y z\n')
                f.write('SIZE 4 4 4\n')
                f.write('TYPE F F F\n')
                f.write('COUNT 1 1 1\n')
                f.write(f'WIDTH {lidar.getHorizontalResolution()}\n')
                f.write(f'HEIGHT {lidar.getNumberOfLayers()}\n')
                f.write(f'POINTS {lidar.getNumberOfPoints()}\n')
                f.write('DATA ascii\n')
                points = lidar.getPointCloud()
                for point in points:
                    f.write(f'{point.x} {point.y} {point.z}\n')

        if (number % 20 == 0):
            # GPS
            dir_name = os.path.join(DATASET_NAME, gps.getName())
            pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(dir_name, index + ".txt"), 'w') as f:
                value = gps.getValues()
                f.write(f'{value[0]} {value[1]} {value[2]}')

        # IMU
        dir_name = os.path.join(DATASET_NAME, IMUx.getName())
        pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dir_name, index + ".txt"), 'w') as f:
            value = IMUx.getRollPitchYaw()
            f.write(f'{value[0]} {value[1]} {value[2]}')

        if (number % 3 == 0):

            # Objects position
            dir_name = os.path.join(DATASET_NAME, 'annotations', 'scene')
            pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(dir_name, index + ".txt"), 'w') as f:
                for object in objects:
                    position = object.getPosition()
                    f.write(f'"{object.getTypeName()}" {position[0]} {position[1]} {position[2]}\n')
                for object in static_objects:
                    position = object['position']
                    type_name = object['type name']
                    f.write(f'"{type_name}" {position[0]} {position[1]} {position[2]}\n')

    def generate_environment(supervisor, field_size):

        root_node = supervisor.getRoot()
        root_children_field = root_node.getField('children')

        if enable_fog:
            fog_node = supervisor.getFromDef('FOG')
            visibility_range = 70.0 * random.randrange(0, 8) + 30
            fog_node.getField('visibilityRange').setSFFloat(visibility_range)

        bkg_index = 0
        print(backgrounds[bkg_index])
        supervisor.getFromDef('BACKGROUND').getField('texture').setSFString(backgrounds[bkg_index])
        supervisor.getFromDef('BACKGROUND_LIGHT').getField('texture').setSFString(backgrounds[bkg_index])

        # add plants
        new_field_node_string = ''
        col = 0
        row = 0
        for i in range(0, field_size[0]):
            row = 0
            for j in range(0, field_size[1]):
                row += 8
            col += 5

        size_x = col * 0.2
        size_y = row * 0.2
        tx = -size_x
        ty = -size_y * 0.5
        new_field_node_string = \
            f"DEF FIELD_SOIL Transform {{ translation {tx} {ty} -0.08 scale 0.2 0.2 0.1 \
            children [ {new_field_node_string} ] }}"
        root_children_field.importMFNodeFromString(-1, new_field_node_string)

        # add field mud
        ground_size_x = size_x + 10
        ground_size_y = size_y + 10
        ground_tx = -size_x / 2
        root_children_field.importMFNodeFromString(-1, f"DEF FIELD_MUD SolidBoxAF {{ \
           translation {ground_tx} 0 0 name \"field mud\" size {ground_size_x} {ground_size_y} 0.1 \
           contactMaterial \"ground\"appearance Soil {{ \
           textureTransform TextureTransform {{ scale 100 100 }} color 0.233341 0.176318 0.112779 }} \
           recognitionColors [ 0.44 0.33 0.24 ] enableBoundingObject TRUE}}")  #

        return {'size': [ground_size_x, ground_size_y], 'translation': [ground_tx, 0]}

    def generate_obstacles(supervisor, objects, obstacle_classes):
        # clean up existing on if needed
        for object in objects:
            object.remove()

        root_node = supervisor.getRoot()
        root_children_field = root_node.getField('children')

        objects = []
        for i in range(0, OBSTACLES_PER_SCENARIO):
            valid = bool(random.getrandbits(1))
            if not valid:
                continue
            index = random.randrange(len(obstacle_classes))
            if index == 0:
                node_name = f"human_{i}"
                model_names = ["Sandra", "Robert", "Anthony", "Sophia"]
                human_index = random.randrange(len(model_names))
                node_string = \
                    f"Robot {{ children [ \
                     CharacterSkin {{ model \"{model_names[human_index]}\" }}\
                    ] \
                    controller \"human_animation\" name \"{node_name}\" model \"{model_names[human_index]}\" \
                    recognitionColors [ 1.0 0.855 0.672 ]}}"
            else:
                parameters = ""
                if obstacle_classes[index] == 'Horse':
                    color_index = random.randrange(3)
                    if color_index == 0:
                        parameters = "colorBody 0.388 0.271 0.173"
                    elif color_index == 1:
                        parameters = "colorBody 0.0733 0.05 0.02"
                    # else default color
                node_string = f'{obstacle_classes[index]} {{ {parameters} }}'
            root_children_field.importMFNodeFromString(-1, node_string)
            objects.append(root_children_field.getMFNode(-1))
        return objects

    def move_object(object, field):
        # randomly set the translation and rotation fields
        pos_x = (random.uniform(0, 1) - 0.5) * field['size'][0] + field['translation'][0]
        pos_y = (random.uniform(0, 1) - 0.5) * field['size'][1] + field['translation'][1]
        translation_field = object.getField('translation')
        translation_field.setSFVec3f([pos_x, pos_y, 0.30])

        angle = random.uniform(0, 1) * 2 * math.pi
        rotation_field = object.getField('rotation')
        rotation_field.setSFRotation([0, 0, 1, angle])

    def move_robot(robot, field, turn_right=False, turn_left=False):
        global already_set_the_velocity

        # set motor veloctiy control
        if (already_set_the_velocity is False):
            leftMotorFront.setPosition(float('inf'))
            leftMotorRear.setPosition(float('inf'))
            rightMotorFront.setPosition(float('inf'))
            rightMotorRear.setPosition(float('inf'))
            leftMotorFront.setVelocity(0.2 * MAX_SPEED)
            leftMotorRear.setVelocity(0.2 * MAX_SPEED)
            rightMotorFront.setVelocity(0.2 * MAX_SPEED)
            rightMotorRear.setVelocity(0.2 * MAX_SPEED)

            already_set_the_velocity = True

        if (turn_right):
            leftMotorFront.setVelocity(0.2 * MAX_SPEED)
            leftMotorRear.setVelocity(0.2 * MAX_SPEED)
            rightMotorFront.setVelocity(0.0 * MAX_SPEED)
            rightMotorRear.setVelocity(0.0 * MAX_SPEED)

            already_set_the_velocity = False

        elif (turn_left):
            leftMotorFront.setVelocity(0.0 * MAX_SPEED)
            leftMotorRear.setVelocity(0.0 * MAX_SPEED)
            rightMotorFront.setVelocity(0.2 * MAX_SPEED)
            rightMotorRear.setVelocity(0.2 * MAX_SPEED)

            already_set_the_velocity = False

    supervisor = Supervisor()
    root_node = supervisor.getRoot()
    root_children_field = root_node.getField('children')

    timestep = int(supervisor.getBasicTimeStep())

    IMUx = supervisor.getInertialUnit('imu_robotti')
    IMUx.enable(timestep)

    # init devices
    # initialize cameras
    cameraBottom = Camera("front_bottom_camera")

    cameras = [cameraBottom]
    for device in cameras:
        device.enable(timestep)
        device.recognitionEnable(timestep)
        device.enableRecognitionSegmentation()
        # device.Recognition.occlusion=False

    # initialize LDIAR
    lidar = Lidar("velodyne")
    lidar.enable(timestep)
    lidar.enablePointCloud()

    # initialize GPS
    gps = GPS("Hemisphere_v500")
    gps.enable(timestep)

    MAX_SPEED = 6.28

    leftMotorFront = supervisor.getDevice("left_front_wheel_joint_motor")
    leftMotorRear = supervisor.getDevice("left_rear_wheel_joint_motor")
    rightMotorFront = supervisor.getDevice("right_front_wheel_joint_motor")
    rightMotorRear = supervisor.getDevice("right_rear_wheel_joint_motor")

    # prepare structures needed for inquiring the scene
    data_index = 1
    categories = []
    obstacle_classes = ['CharacterSkin', 'Cat', 'Cow', 'Deer', 'Dog', 'Fox', 'Horse', 'Rabbit', 'Sheep']

    # retrieve static objects
    static_objects = []
    for i in range(0, root_children_field.getCount()):
        child = root_children_field.getMFNode(i)
        object_classes = ['AgriculturalWarehouse', 'Barn', 'Tractor', 'BungalowStyleHouse', 'HouseWithGarage', 'Silo',
                          'SimpleTree',
                          'Forest', 'PicketFenceWithDoor', 'PicketFence', 'Forest', 'Road', 'StraightRoadSegment',
                          'RoadIntersection']
        object_classes += obstacle_classes
        if child.getTypeName() in object_classes:
            static_objects.append({'type name': child.getTypeName(), 'position': child.getPosition()})

    import time

    objects = []
    last_yaw = 1.57
    range_image_rgb = ''

    # generate scenation
    field = generate_environment(supervisor, FIELD_SIZE)

    objects = generate_obstacles(supervisor, objects, obstacle_classes)

    for object in objects:
        if object.getTypeName() not in categories:
            categories.append(object.getTypeName())

    supervisor.step(1 * timestep)
    start_time = time.time()

    # main loop
    index = 1
    records_per_second = 0
    second = 1

    # move objects in the field
    for object in objects:
        move_object(object, field)

    while index < MAX_RECORDS_PER_SCENARIO:
        current_time = time.time()
        if (current_time - start_time < 1):
            records_per_second += 1
        else:
            records_per_second = 0
            start_time = time.time()

        value = gps.getValues()

        roll_pitch_yaw_values = IMUx.getRollPitchYaw()

        object_list = []
        for object in device.getRecognitionObjects():
            object_list.append(object.getModel())

        # move robot in the field
        if (value[0] < -45 and roll_pitch_yaw_values[2] > 0):
            move_robot(supervisor.getSelf(), field, True, False)
            last_yaw = 0

        elif (value[0] < -43.0 and value[1] > -12.29 and roll_pitch_yaw_values[2] > -1.57):
            move_robot(supervisor.getSelf(), field, True, False)

            last_yaw = -1.57

        elif (value[0] > 2.0 and roll_pitch_yaw_values[2] < 0):
            move_robot(supervisor.getSelf(), field, False, True)

            last_yaw = 0

        elif (value[0] > 0.0 and roll_pitch_yaw_values[2] < 1.57):
            move_robot(supervisor.getSelf(), field, False, True)

            last_yaw = 1.57

        else:
            if (roll_pitch_yaw_values[2] > last_yaw + 0.01):
                move_robot(supervisor.getSelf(), field, True, False)

            elif (roll_pitch_yaw_values[2] < last_yaw - 0.01):
                move_robot(supervisor.getSelf(), field, False, True)

            else:
                move_robot(supervisor.getSelf(), field, False, False)

        if supervisor.step(timestep) == -1:
            exit()

        # save sensor measurements
        save_device_measurements(str(data_index), str(second), cameras, lidar, gps, objects, static_objects)

        data_index += 1
        index += 1

        if (index % 100 == 0):

            print(f'Robotti second: {second}')

            second += 1
            index = 1
            if (second == STOP_ON):
                index = MAX_RECORDS_PER_SCENARIO

    save_datasets_info(index - 1, categories)
    print('Completed.')

    leftMotorFront.setVelocity(0.0 * MAX_SPEED)
    leftMotorRear.setVelocity(0.0 * MAX_SPEED)
    rightMotorFront.setVelocity(0.0 * MAX_SPEED)
    rightMotorRear.setVelocity(0.0 * MAX_SPEED)

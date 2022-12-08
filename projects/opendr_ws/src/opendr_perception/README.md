# Perception Package

This package contains ROS nodes related to perception package of OpenDR.

## Dataset ROS Nodes

Assuming that you have already [built your workspace](../../README.md) and started roscore (i.e., just run `roscore`), then you can start a dataset node to publish data from the disk, which is useful to test the functionality without the use of a sensor.
Dataset nodes take a `DatasetIterator` object that shoud returns a `(Data, Target)` pair elements.
If the type of the `Data` object is correct, the node will transform it into a corresponding ROS message object and publish it to a desired topic.

### Point Cloud Dataset ROS Node
To get a point cloud from a dataset on the disk, you can start a `point_cloud_dataset.py` node as:
```shell
rosrun perception point_cloud_dataset.py
```
By default, it downloads a `nano_KITTI` dataset from OpenDR's FTP server and uses it to publish data to the ROS topic. You can create an instance of this node with any `DatasetIterator` object that returns `(PointCloud, Target)` as elements.

### Image Dataset ROS Node
To get an image from a dataset on the disk, you can start a `image_dataset.py` node as:
```shell
rosrun perception image_dataset.py
```
By default, it downloads a `nano_MOT20` dataset from OpenDR's FTP server and uses it to publish data to the ROS topic. You can create an instance of this node with any `DatasetIterator` object that returns `(Image, Target)` as elements.

## Pose Estimation ROS Node
Assuming that you have already [activated the OpenDR environment](../../../../docs/reference/installation.md), [built your workspace](../../README.md) and started roscore (i.e., just run `roscore`), then you can

1. Start the node responsible for publishing images. If you have a usb camera, then you can use the corresponding node (assuming you have installed the corresponding package):

```shell
rosrun usb_cam usb_cam_node
```

2. You are then ready to start the pose detection node (use `-h` to print out help for various arguments)

```shell
rosrun perception pose_estimation.py
```

3. You can examine the annotated image stream using `rqt_image_view` (select the topic `/opendr/image_pose_annotated`) or
   `rostopic echo /opendr/poses`.

Note that to use the pose messages properly, you need to create an appropriate subscriber that will convert the ROS pose messages back to OpenDR poses which you can access as described in the [documentation](https://github.com/opendr-eu/opendr/blob/master/docs/reference/engine-target.md#posekeypoints-confidence):
```python
        ...
        rospy.Subscriber("opendr/poses", Detection2DArray, self.callback)
        ...
        def callback(self, data):
            opendr_pose = self.bridge.from_ros_pose(data)
            print(opendr_pose)
            print(opendr_pose['r_eye'])
```

## Fall Detection ROS Node
Assuming that you have already [activated the OpenDR environment](../../../../docs/reference/installation.md), [built your workspace](../../README.md) and started roscore (i.e., just run `roscore`), then you can

1. Start the node responsible for publishing images. If you have a usb camera, then you can use the corresponding node (assuming you have installed the corresponding package):

```shell
rosrun usb_cam usb_cam_node
```

2. You are then ready to start the fall detection node

```shell
rosrun perception fall_detection.py
```

3. You can examine the annotated image stream using `rqt_image_view` (select the topic `/opendr/image_fall_annotated`) or
   `rostopic echo /opendr/falls`, where the node publishes bounding boxes of detected fallen poses

## Face Recognition ROS Node
Assuming that you have already [activated the OpenDR environment](../../../../docs/reference/installation.md), [built your workspace](../../README.md) and started roscore (i.e., just run `roscore`), then you can


1. Start the node responsible for publishing images. If you have a usb camera, then you can use the corresponding node (assuming you have installed the corresponding package):

```shell
rosrun usb_cam usb_cam_node
```

2. You are then ready to start the face recognition node. Note that you should pass the folder containing the images of known faces as argument to create the corresponding database of known persons.

```shell
rosrun perception face_recognition.py _database_path:='./database'
```
**Notes**

Reference images should be placed in a defined structure like:
- imgs
    - ID1
      - image1
      - image2
    - ID2
    - ID3
    - ...

Τhe name of the sub-folder, e.g. ID1, will be published under `/opendr/face_recognition_id`.

4. The database entry and the returned confidence is published under the topic name `/opendr/face_recognition`, and the human-readable ID
under `/opendr/face_recognition_id`.

## 2D Object Detection ROS Nodes
ROS nodes are implemented for the SSD, YOLOv3, CenterNet, DETR, Nanodet and YOLOv5 generic object detectors.
Assuming that you have already [activated the OpenDR environment](../../../../docs/reference/installation.md), [built your workspace](../../README.md) and started roscore (i.e., just run `roscore`).

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the corresponding node (assuming you have installed the corresponding package):
```shell
rosrun usb_cam usb_cam_node
```

2. Then, to initiate the SSD detector node, run:

```shell
rosrun perception object_detection_2d_ssd.py
```
The annotated image stream can be viewed using `rqt_image_view`, and the default topic name is
`/opendr/image_boxes_annotated`. The bounding boxes alone are also published as `/opendr/objects`.
Similarly, the YOLOv3, CenterNet, DETR, Nanodet and YOLOv5 detector nodes can be run with:
```shell
rosrun perception object_detection_2d_yolov3.py
```
or
```shell
rosrun perception object_detection_2d_centernet.py
```
or
```shell
rosrun perception object_detection_2d_detr.py
```
or
```shell
rosrun perception object_detection_2d_nanodet.py
```
or
```shell
rosrun perception object_detection_2d_yolov5.py
```
respectively.

## Face Detection ROS Node
A ROS node for the RetinaFace detector is implemented, supporting both the ResNet and MobileNet versions, the latter of
which performs mask recognition as well. After setting up the environment, the detector node can be initiated as:
```shell
rosrun perception face_detection_retinaface.py
```
The annotated image stream is published under the topic name `/opendr/image_boxes_annotated`, and the bounding boxes alone
under `/opendr/faces`.

## GEM ROS Node
Assuming that you have already [built your workspace](../../README.md) and started roscore (i.e., just run `roscore`), then you can


1. Add OpenDR to `PYTHONPATH` (please make sure you do not overwrite `PYTHONPATH` ), e.g.,
```shell
export PYTHONPATH="/home/user/opendr/src:$PYTHONPATH"
```
2. First one needs to find points in the color and infrared images that correspond, in order to find the homography matrix that allows to correct for the difference in perspective between the infrared and the RGB camera.
These points can be selected using a [utility tool](../../../../src/opendr/perception/object_detection_2d/utils/get_color_infra_alignment.py) that is provided in the toolkit.

3. Pass the points you have found as *pts_color* and *pts_infra* arguments to the ROS gem.py node.

4. Start the node responsible for publishing images. If you have a RealSense camera, then you can use the corresponding node (assuming you have installed [realsense2_camera](http://wiki.ros.org/realsense2_camera)):

```shell
roslaunch realsense2_camera rs_camera.launch enable_color:=true enable_infra:=true enable_depth:=false enable_sync:=true infra_width:=640 infra_height:=480
```

4. You are then ready to start the pose detection node

```shell
rosrun perception object_detection_2d_gem.py
```

5. You can examine the annotated image stream using `rqt_image_view` (select one of the topics `/opendr/color_detection_annotated` or `/opendr/infra_detection_annotated`) or `rostopic echo /opendr/detections`


## Panoptic Segmentation ROS Node
A ROS node for performing panoptic segmentation on a specified RGB image stream using the [EfficientPS](../../../../src/opendr/perception/panoptic_segmentation/README.md) network.
Assuming that the OpenDR catkin workspace has been sourced, the node can be started with:
```shell
rosrun perception panoptic_segmentation_efficient_ps.py
```

The following optional arguments are available:
- `-h, --help`: show a help message and exit
- `--input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC` : listen to RGB images on this topic (default=`/usb_cam/image_raw`)
- `--checkpoint CHECKPOINT` : download pretrained models [cityscapes, kitti] or load from the provided path (default=`cityscapes`)
- `--output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: publish the semantic and instance maps on this topic as `OUTPUT_HEATMAP_TOPIC/semantic` and `OUTPUT_HEATMAP_TOPIC/instance` (default=`/opendir/panoptic`)
- `--visualization_topic VISUALIZATION_TOPIC`: publish the panoptic segmentation map as an RGB image on `VISUALIZATION_TOPIC` or a more detailed overview if using the `--detailed_visualization` flag (default=`/opendr/panoptic/rgb_visualization`)
- `--detailed_visualization`: generate a combined overview of the input RGB image and the semantic, instance, and panoptic segmentation maps and publish it on `OUTPUT_RGB_IMAGE_TOPIC` (default=deactivated)


## Semantic Segmentation ROS Node
A ROS node for performing semantic segmentation on an input image using the BiseNet model.
Assuming that the OpenDR catkin workspace has been sourced, the node can be started with:
```shell
rosrun perception semantic_segmentation_bisenet.py IMAGE_TOPIC
```

Additionally, the following optional arguments are available:
- `-h, --help`: show a help message and exit
- `--heamap_topic HEATMAP_TOPIC`: publish the heatmap on `HEATMAP_TOPIC`

## RGBD Hand Gesture Recognition ROS Node

A ROS node for performing hand gesture recognition using MobileNetv2 model trained on HANDS dataset. The node has been tested with Kinectv2 for depth data acquisition with the following drivers: https://github.com/OpenKinect/libfreenect2 and https://github.com/code-iai/iai_kinect2. Assuming that the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception rgbd_hand_gesture_recognition.py
```
The predictied classes are published to the topic `/opendr/gestures`.

## Heart Anomaly Detection ROS Node

A ROS node for performing heart anomaly (atrial fibrillation) detection from ecg data using GRU or ANBOF models trained on AF dataset. Assuming that the OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception heart_anomaly_detection.py ECG_TOPIC MODEL
```
with `ECG_TOPIC` specifying the ROS topic to which the node will subscribe, and `MODEL` set to either *gru* or *anbof*. The predicted classes are published to the topic `/opendr/heartanomaly`.

## Human Action Recognition ROS Node

A ROS node for performing Human Activity Recognition using either CoX3D or X3D models pretrained on Kinetics400.
Assuming the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception video_activity_recognition.py
```
The predicted class id and confidence is published under the topic name `/opendr/human_activity_recognition`, and the human-readable class name under `/opendr/human_activity_recognition_description`.

## Landmark-based Facial Expression Recognition ROS Node

A ROS node for performing Landmark-based Facial Expression Recognition using the pretrained model PST-BLN on AFEW, CK+ or Oulu-CASIA datasets.
Assuming the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception landmark_based_facial_expression_recognition.py
```
The predicted class id and confidence is published under the topic name `/opendr/landmark_based_expression_recognition`, and the human-readable class name under `/opendr/landmark_based_expression_recognition_description`.

## Skeleton-based Human Action Recognition ROS Node

A ROS node for performing Skeleton-based Human Action Recognition using either ST-GCN or PST-GCN models pretrained on NTU-RGBD-60 dataset. The human body poses of the image are first extracted by the light-weight Open-pose method which is implemented in the toolkit, and they are passed to the skeleton-based action recognition method to be categorized.
Assuming the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception skeleton_based_action_recognition.py
```
The predicted class id and confidence is published under the topic name `/opendr/skeleton_based_action_recognition`, and the human-readable class name under `/opendr/skeleton_based_action_recognition_description`.
Besides, the annotated image is published in `/opendr/image_pose_annotated` as well as the corresponding poses in `/opendr/poses`.

## Speech Command Recognition ROS Node

A ROS node for recognizing speech commands from an audio stream using MatchboxNet, EdgeSpeechNets or Quadratic SelfONN models, pretrained on the Google Speech Commands dataset.
Assuming that the OpenDR catkin workspace has been sourced, the node can be started with:
```shell
rosrun perception speech_command_recognition.py INPUT_AUDIO_TOPIC
```
The following optional arguments are available:
- `--buffer_size BUFFER_SIZE`: set the size of the audio buffer (expected command duration) in seconds, default value **1.5**
- `--model MODEL`: choose the model to use: `matchboxnet` (default value), `edgespeechnets` or `quad_selfonn`
- `--model_path MODEL_PATH`: if given, the pretrained model will be loaded from the specified local path, otherwise it will be downloaded from an OpenDR FTP server

The predictions (class id and confidence) are published to the topic `/opendr/speech_recognition`.
**Note:** EdgeSpeechNets currently does not have a pretrained model available for download, only local files may be used.

## Voxel Object Detection 3D ROS Node

A ROS node for performing Object Detection 3D using PointPillars or TANet methods with either pretrained models on KITTI dataset, or custom trained models.
The predicted detection annotations are pushed to `output_detection3d_topic` (default `output_detection3d_topic="/opendr/detection3d"`).

Assuming the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception object_detection_3d_voxel.py
```
To get a point cloud from a dataset on the disk, you can start a `point_cloud_dataset.py` node as:
```shell
rosrun perception point_cloud_dataset.py
```
This will publish the dataset point clouds to a `/opendr/dataset_point_cloud` topic by default, which means that the `input_point_cloud_topic` should be set to `/opendr/dataset_point_cloud`.

## AB3DMOT Object Tracking 3D ROS Node

A ROS node for performing Object Tracking 3D using AB3DMOT stateless method.
This is a detection-based method, and therefore the 3D object detector is needed to provide detections, which then will be used to make associations and generate tracking ids.
The predicted tracking annotations are split into two topics with detections (default `output_detection_topic="/opendr/detection3d"`) and tracking ids (default `output_tracking_id_topic="/opendr/tracking3d_id"`).

Assuming the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception object_tracking_3d_ab3dmot.py
```
To get a point cloud from a dataset on the disk, you can start a `point_cloud_dataset.py` node as:
```shell
rosrun perception point_cloud_dataset.py
```
This will publish the dataset point clouds to a `/opendr/dataset_point_cloud` topic by default, which means that the `input_point_cloud_topic` should be set to `/opendr/dataset_point_cloud`.


## FairMOT Object Tracking 2D ROS Node

A ROS node for performing Object Tracking 2D using FairMOT with either pretrained models on MOT dataset, or custom trained models. The predicted tracking annotations are split into two topics with detections (default `output_detection_topic="/opendr/detection"`) and tracking ids (default `output_tracking_id_topic="/opendr/tracking_id"`). Additionally, an annotated image is generated if the `output_image_topic` is not None (default `output_image_topic="/opendr/image_annotated"`)
Assuming the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception object_tracking_2d_fair_mot.py
```
To get images from usb_camera, you can start the camera node as:
```shell
rosrun usb_cam usb_cam_node
```
The corresponding `input_image_topic` should be `/usb_cam/image_raw`.
If you want to use a dataset from the disk, you can start a `image_dataset.py` node as:
```shell
rosrun perception image_dataset.py
```
This will publish the dataset images to an `/opendr/dataset_image` topic by default, which means that the `input_image_topic` should be set to `/opendr/dataset_image`.

## Deep Sort Object Tracking 2D ROS Node

A ROS node for performing Object Tracking 2D using Deep Sort using either pretrained models on Market1501 dataset, or custom trained models. This is a detection-based method, and therefore the 2D object detector is needed to provide detections, which then will be used to make associations and generate tracking ids. The predicted tracking annotations are split into two topics with detections (default `output_detection_topic="/opendr/detection"`) and tracking ids (default `output_tracking_id_topic="/opendr/tracking_id"`). Additionally, an annotated image is generated if the `output_image_topic` is not None (default `output_image_topic="/opendr/image_annotated"`)
Assuming the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception object_tracking_2d_deep_sort.py
```
To get images from usb_camera, you can start the camera node as:
```shell
rosrun usb_cam usb_cam_node
```
The corresponding `input_image_topic` should be `/usb_cam/image_raw`.
If you want to use a dataset from the disk, you can start an `image_dataset.py` node as:
```shell
rosrun perception image_dataset.py
```
This will publish the dataset images to an `/opendr/dataset_image` topic by default, which means that the `input_image_topic` should be set to `/opendr/dataset_image`.


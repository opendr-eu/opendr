# OpenDR Perception Package

This package contains ROS2 nodes related to the perception package of OpenDR.

---

## Prerequisites

Before you can run any of the toolkit's ROS2 nodes, some prerequisites need to be fulfilled:
1. First of all, you need to [set up the required packages and build your workspace.](../../README.md#first-time-setup)
2. _(Optional for nodes with [RGB input](#rgb-input-nodes))_ 

    For basic usage and testing, all the toolkit's ROS2 nodes that use RGB images are set up to expect input from a basic webcam using the default package `usb_cam` which is installed with OpenDR. You can run the webcam node in a new terminal:
    ```shell
    ros2 run usb_cam usb_cam_node_exe
    ```
    By default, the USB cam node publishes images on `/image_raw` and the RGB input nodes subscribe to this topic if not provided with an input topic argument. 
    As explained for each node below, you can modify the topics via arguments, so if you use any other node responsible for publishing images, **make sure to change the input topic accordingly.**

3. _(Optional for nodes with [audio input](#audio-input) or [audiovisual input](#rgb--audio-input))_
    
    For basic usage and testing, the toolkit's ROS2 nodes that use audio as input are set up to expect input from a basic audio device using the default package `audio_common`  which is installed with OpenDR. You can run the audio node in a new terminal:
    ```shell
    ros2 launch audio_capture capture_wave.launch.xml
    ```
    By default, the audio capture node publishes audio data on `/audio/audio` and the audio input nodes subscribe to this topic if not provided with an input topic argument. 
    As explained for each node below, you can modify the topics via arguments, so if you use any other node responsible for publishing audio, **make sure to change the input topic accordingly.**

---

## Notes

- ### Display output images with rqt_image_view
    For any node that outputs images, `rqt_image_view` can be used to display them by running the following command:
    ```shell
    ros2 run rqt_image_view rqt_image_view &
    ```
    A window will appear, where the topic that you want to view can be selected from the drop-down menu on the top-left area of the window.
    Refer to each node's documentation below to find out the default output image topic, where applicable, and select it on the drop-down menu of rqt_image_view.

- ### Echo node output
    All OpenDR nodes publish some kind of detection message, which can be echoed by running the following command:
    ```shell
    ros2 topic echo /opendr/topic_name
    ```
    You can find out the default topic name for each node, in its documentation below.

- ### Increase performance by disabling output
    Optionally, nodes can be modified via command line arguments, which are presented for each node separately below.
    Generally, arguments give the option to change the input and output topics, the device the node runs on (CPU or GPU), etc.
    When a node publishes on several topics, where applicable, a user can opt to disable one or more of the outputs by providing `None` in the corresponding output topic.
    This disables publishing on that topic, forgoing some operations in the node, which might increase its performance.

    _An example would be to disable the output annotated image topic in a node when visualization is not needed and only use the detection message in another node, thus eliminating the OpenCV operations._

- ### Logging the node performance in the console
   OpenDR provides the utility [performance node](#performance-ros2-node) to log performance messages in the console for the running node.
   You can set the `performance_topic` of the node you are using and also run the performance node to get the time it takes for the
   node to process a single input and its average speed expressed in frames per second.

- ### An example diagram of OpenDR nodes running
    ![Face Detection ROS2 node running diagram](../../images/opendr_node_diagram.png)
    - On the left, the `usb_cam` node can be seen, which is using a system camera to publish images on the `/image_raw` topic.
    - In the middle, OpenDR's face detection node is running taking as input the published image. By default, the node has its input topic set to `/image_raw`.
    - To the right the two output topics of the face detection node can be seen.
    The bottom topic `/opendr/image_faces_annotated` is the annotated image which can be easily viewed with `rqt_image_view` as explained earlier.
    The other topic `/opendr/faces` is the detection message which contains the detected faces' detailed information.
    This message can be easily viewed by running `ros2 topic echo /opendr/faces` in a terminal.

<!-- - ### Other notes -->

----

## RGB input nodes

### Pose Estimation ROS2 Node

You can find the pose estimation ROS2 node python script [here](./opendr_perception/pose_estimation_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [pose estimation tool](../../../../src/opendr/perception/pose_estimation/lightweight_open_pose/lightweight_open_pose_learner.py) whose documentation can be found [here](../../../../docs/reference/lightweight-open-pose.md).
The node publishes the detected poses in [OpenDR's 2D pose message format](../opendr_interface/msg/OpenDRPose2D.msg), which saves a list of [OpenDR's keypoint message format](../opendr_interface/msg/OpenDRPose2DKeypoint.msg).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the pose detection node:
    ```shell
    ros2 run opendr_perception pose_estimation
    ```
    The following optional arguments are available:
   - `-h, --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_pose_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages, `None` to stop the node from publishing on this topic (default=`/opendr/poses`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: Device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `--accelerate`: Acceleration flag that causes pose estimation to run faster but with less accuracy

3. Default output topics:
   - Output images: `/opendr/image_pose_annotated`
   - Detection messages: `/opendr/poses`

   For viewing the output, refer to the [notes above.](#notes)

### High Resolution Pose Estimation ROS2 Node

You can find the high resolution pose estimation ROS2 node python script [here](./opendr_perception/hr_pose_estimation_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [high resolution pose estimation tool](../../../../src/opendr/perception/pose_estimation/hr_pose_estimation/high_resolution_learner.py) whose documentation can be found [here](../../../../docs/reference/high-resolution-pose-estimation.md).
The node publishes the detected poses in [OpenDR's 2D pose message format](../opendr_interface/msg/OpenDRPose2D.msg), which saves a list of [OpenDR's keypoint message format](../opendr_interface/msg/OpenDRPose2DKeypoint.msg).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the high resolution pose detection node:
    ```shell
    ros2 run opendr_perception hr_pose_estimation
    ```
    The following optional arguments are available:
   - `-h, --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_pose_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages, `None` to stop the node from publishing on this topic (default=`/opendr/poses`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: Device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `--accelerate`: Acceleration flag that causes pose estimation to run faster but with less accuracy

3. Default output topics:
   - Output images: `/opendr/image_pose_annotated`
   - Detection messages: `/opendr/poses`

   For viewing the output, refer to the [notes above.](#notes)

### Fall Detection ROS2 Node

You can find the fall detection ROS2 node python script [here](./opendr_perception/fall_detection_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [fall detection tool](../../../../src/opendr/perception/fall_detection/fall_detector_learner.py) whose documentation can be found [here](../../../../docs/reference/fall-detection.md).
Fall detection is rule-based and works on top of pose estimation.

This node normally runs on `detection mode` where it subscribes to a topic of OpenDR poses and detects whether the poses are fallen persons or not.
By providing an image topic the node runs on `visualization mode`. It also gets images, performs pose estimation internally and visualizes the output on an output image topic.
Note that when providing an image topic the node has significantly worse performance in terms of speed, due to running pose estimation internally.

- #### Instructions for basic usage in `detection mode`:

1. Start the node responsible for publishing poses. Refer to the [pose estimation node above](#pose-estimation-ros2-node).

2. You are then ready to start the fall detection node:

    ```shell
    ros2 run opendr_perception fall_detection
    ```
   The following optional arguments are available and relevant for running fall detection on pose messages only:
   - `-h or --help`: show a help message and exit
   - `-ip or --input_pose_topic INPUT_POSE_TOPIC`: topic name for input pose, `None` to stop the node from running detections on pose messages (default=`/opendr/poses`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages (default=`/opendr/fallen`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages, note that performance will be published to `PERFORMANCE_TOPIC/fallen` (default=`None`, disabled)

3. Detections are published on the `detections_topic`

- #### Instructions for `visualization mode`:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the fall detection node in `visualization mode`, which needs an input image topic to be provided:

    ```shell
    ros2 run opendr_perception fall_detection -ii /image_raw
    ```
    The following optional arguments are available and relevant for running fall detection on images. Note that the
`input_rgb_image_topic` is required for running in `visualization mode`:
   - `-h or --help`: show a help message and exit
   - `-ii or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`None`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image (default=`/opendr/image_fallen_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages (default=`/opendr/fallen`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages, note that performance will be published to `PERFORMANCE_TOPIC/image` (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `--accelerate`: acceleration flag that causes pose estimation that runs internally to run faster but with less accuracy


- Default output topics:
  - Detection messages: `/opendr/fallen`
  - Output images: `/opendr/image_fallen_annotated`

  For viewing the output, refer to the [notes above.](#notes)

**Notes**

Note that when the node runs on the default `detection mode` it is significantly faster than when it is provided with an 
input image topic. However, pose estimation needs to be performed externally on another node which publishes poses.
When an input image topic is provided and the node runs in `visualization mode`, it runs pose estimation internally, and 
consequently it is recommended to only use it for testing purposes and not run other pose estimation nodes in parallel.
The node can run in both modes in parallel or only on one of the two. To run the node only on `visualization mode` provide
the argument `-ip None` to disable the `detection mode`. Detection messages on `detections_topic` are published in both modes.

### Wave Detection ROS2 Node

You can find the wave detection ROS2 node python script [here](./opendr_perception/wave_detection_node.py) to inspect the code and modify it as you wish to fit your needs.
The node is based on a [wave detection demo of the Lightweight OpenPose tool](../../../../projects/python/perception/pose_estimation/lightweight_open_pose/demos/wave_detection_demo.py).
Wave detection is rule-based and works on top of pose estimation.

This node normally runs on `detection mode` where it subscribes to a topic of OpenDR poses and detects whether the poses are waving or not.
By providing an image topic the node runs on `visualization mode`. It also gets images, performs pose estimation internally and visualizes the output on an output image topic.
Note that when providing an image topic the node has significantly worse performance in terms of speed, due to running pose estimation internally.

- #### Instructions for basic usage in `detection mode`:

1. Start the node responsible for publishing poses. Refer to the [pose estimation node above](#pose-estimation-ros2-node).

2. You are then ready to start the wave detection node:

    ```shell
    ros2 run opendr_perception wave_detection
    ```
    The following optional arguments are available and relevant for running fall detection on pose messages only:
   - `-h or --help`: show a help message and exit
   - `-ip or --input_pose_topic INPUT_POSE_TOPIC`: topic name for input pose, `None` to stop the node from running detections on pose messages (default=`/opendr/poses`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages (default=`/opendr/wave`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages, note that performance will be published to `PERFORMANCE_TOPIC/wave` (default=`None`, disabled)

3. Detections are published on the `detections_topic`

- #### Instructions for `visualization mode`:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the wave detection node in `visualization mode`, which needs an input image topic to be provided:

    ```shell
    ros2 run opendr_perception wave_detection -ii /image_raw
    ```
    The following optional arguments are available and relevant for running wave detection on images. Note that the
`input_rgb_image_topic` is required for running in `visualization mode`:
   - `-h or --help`: show a help message and exit
   - `-ii or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`None`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image (default=`/opendr/image_wave_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages (default=`/opendr/wave`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages, note that performance will be published to `PERFORMANCE_TOPIC/image` (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `--accelerate`: acceleration flag that causes pose estimation that runs internally to run faster but with less accuracy

- Default output topics:
  - Detection messages: `/opendr/wave`
  - Output images: `/opendr/image_wave_annotated`

  For viewing the output, refer to the [notes above.](#notes)

**Notes**

Note that when the node runs on the default `detection mode` it is significantly faster than when it is provided with an 
input image topic. However, pose estimation needs to be performed externally on another node which publishes poses.
When an input image topic is provided and the node runs in `visualization mode`, it runs pose estimation internally, and 
consequently it is recommended to only use it for testing purposes and not run other pose estimation nodes in parallel.
The node can run in both modes in parallel or only on one of the two. To run the node only on `visualization mode` provide
the argument `-ip None` to disable the `detection mode`. Detection messages on `detections_topic` are published in both modes.

### Face Detection ROS2 Node

The face detection ROS2 node supports both the ResNet and MobileNet versions, the latter of which performs masked face detection as well.

You can find the face detection ROS2 node python script [here](./opendr_perception/face_detection_retinaface_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [face detection tool](../../../../src/opendr/perception/object_detection_2d/retinaface/retinaface_learner.py) whose documentation can be found [here](../../../../docs/reference/face-detection-2d-retinaface.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the face detection node

    ```shell
    ros2 run opendr_perception face_detection_retinaface
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_faces_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages, `None` to stop the node from publishing on this topic (default=`/opendr/faces`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `--backbone BACKBONE`: retinaface backbone, options are either `mnet` or `resnet`, where `mnet` detects masked faces as well (default=`resnet`)

3. Default output topics:
   - Output images: `/opendr/image_faces_annotated`
   - Detection messages: `/opendr/faces`

   For viewing the output, refer to the [notes above.](#notes)

### Face Recognition ROS2 Node

You can find the face recognition ROS2 node python script [here](./opendr_perception/face_recognition_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [face recognition tool](../../../../src/opendr/perception/face_recognition/face_recognition_learner.py) whose documentation can be found [here](../../../../docs/reference/face-recognition.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the face recognition node:

    ```shell
    ros2 run opendr_perception face_recognition
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_face_reco_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages, `None` to stop the node from publishing on this topic (default=`/opendr/face_recognition`)
   - `-id or --detections_id_topic DETECTIONS_ID_TOPIC`: topic name for detection ID messages, `None` to stop the node from publishing on this topic (default=`/opendr/face_recognition_id`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `--backbone BACKBONE`: backbone network (default=`mobilefacenet`)
   - `--dataset_path DATASET_PATH`: path of the directory where the images of the faces to be recognized are stored (default=`./database`)

3. Default output topics:
   - Output images: `/opendr/image_face_reco_annotated`
   - Detection messages: `/opendr/face_recognition` and `/opendr/face_recognition_id`

   For viewing the output, refer to the [notes above.](#notes)

**Notes**

Reference images should be placed in a defined structure like:
- imgs
    - ID1
      - image1
      - image2
    - ID2
    - ID3
    - ...

The default dataset path is `./database`. Please use the `--database_path ./your/path/` argument to define a custom one.
Τhe name of the sub-folder, e.g. ID1, will be published under `/opendr/face_recognition_id`.

The database entry and the returned confidence is published under the topic name `/opendr/face_recognition`, and the human-readable ID
under `/opendr/face_recognition_id`.

### 2D Object Detection ROS2 Nodes

For 2D object detection, there are several ROS2 nodes implemented using various algorithms. The generic object detectors are SSD, YOLOv3, YOLOv5, CenterNet, Nanodet and DETR.

You can find the 2D object detection ROS2 node python scripts here:
[SSD node](./opendr_perception/object_detection_2d_ssd_node.py), [YOLOv3 node](./opendr_perception/object_detection_2d_yolov3_node.py), [YOLOv5 node](./opendr_perception/object_detection_2d_yolov5_node.py), [CenterNet node](./opendr_perception/object_detection_2d_centernet_node.py), [Nanodet node](./opendr_perception/object_detection_2d_nanodet_node.py) and [DETR node](./opendr_perception/object_detection_2d_detr_node.py),
where you can inspect the code and modify it as you wish to fit your needs.
The nodes makes use of the toolkit's various 2D object detection tools:
[SSD tool](../../../../src/opendr/perception/object_detection_2d/ssd/ssd_learner.py), [YOLOv3 tool](../../../../src/opendr/perception/object_detection_2d/yolov3/yolov3_learner.py), [YOLOv5 tool](../../../../src/opendr/perception/object_detection_2d/yolov5/yolov5_learner.py),
[CenterNet tool](../../../../src/opendr/perception/object_detection_2d/centernet/centernet_learner.py), [Nanodet tool](../../../../src/opendr/perception/object_detection_2d/nanodet/nanodet_learner.py), [DETR tool](../../../../src/opendr/perception/object_detection_2d/detr/detr_learner.py),
whose documentation can be found here:
[SSD docs](../../../../docs/reference/object-detection-2d-ssd.md), [YOLOv3 docs](../../../../docs/reference/object-detection-2d-yolov3.md), [YOLOv5 docs](../../../../docs/reference/object-detection-2d-yolov5.md),
[CenterNet docs](../../../../docs/reference/object-detection-2d-centernet.md), [Nanodet docs](../../../../docs/reference/nanodet.md), [DETR docs](../../../../docs/reference/detr.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start a 2D object detector node:
   1. SSD node
      ```shell
      ros2 run opendr_perception object_detection_2d_ssd
      ```
      The following optional arguments are available for the SSD node:
      - `--backbone BACKBONE`: Backbone network (default=`vgg16_atrous`)
      - `--nms_type NMS_TYPE`: Non-Maximum Suppression type options are `default`, `seq2seq-nms`, `soft-nms`, `fast-nms`, `cluster-nms` (default=`default`)

   2. YOLOv3 node
      ```shell
      ros2 run opendr_perception object_detection_2d_yolov3
      ```
      The following optional argument is available for the YOLOv3 node:
      - `--backbone BACKBONE`: Backbone network (default=`darknet53`)

   3. YOLOv5 node
      ```shell
      ros2 run opendr_perception object_detection_2d_yolov5
      ```
      The following optional argument is available for the YOLOv5 node:
      - `--model_name MODEL_NAME`: Network architecture, options are `yolov5s`, `yolov5n`, `yolov5m`, `yolov5l`, `yolov5x`, `yolov5n6`, `yolov5s6`, `yolov5m6`, `yolov5l6`, `custom` (default=`yolov5s`)

   4. CenterNet node
      ```shell
      ros2 run opendr_perception object_detection_2d_centernet
      ```
      The following optional argument is available for the CenterNet node:
      - `--backbone BACKBONE`: Backbone network (default=`resnet50_v1b`)

   5. Nanodet node
      ```shell
      ros2 run opendr_perception object_detection_2d_nanodet
      ```
      The following optional argument is available for the Nanodet node:
      - `--model Model`: Model that config file will be used (default=`plus_m_1.5x_416`)

   6. DETR node
      ```shell
      ros2 run opendr_perception object_detection_2d_detr
      ```

   The following optional arguments are available for all nodes above:
   - `-h or --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_objects_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages, `None` to stop the node from publishing on this topic (default=`/opendr/objects`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: Device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)

3. Default output topics:
   - Output images: `/opendr/image_objects_annotated`
   - Detection messages: `/opendr/objects`

   For viewing the output, refer to the [notes above.](#notes)

### 2D Single Object Tracking ROS2 Node

You can find the single object tracking 2D ROS2 node python script [here](./opendr_perception/object_tracking_2d_siamrpn_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [single object tracking 2D SiamRPN tool](../../../../src/opendr/perception/object_tracking_2d/siamrpn/siamrpn_learner.py) whose documentation can be found [here](../../../../docs/reference/object-tracking-2d-siamrpn.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the single object tracking 2D node:

    ```shell
    ros2 run opendr_perception object_tracking_2d_siamrpn
    ```

    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC` : listen to RGB images on this topic (default=`/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_tracking_annotated`)
   - `-t or --tracker_topic TRACKER_TOPIC`: topic name for tracker messages, `None` to stop the node from publishing on this topic (default=`/opendr/tracked_object`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: Device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)

3. Default output topics:
   - Output images: `/opendr/image_tracking_annotated`
   - Detection messages: `/opendr/tracked_object`

   For viewing the output, refer to the [notes above.](#notes)

**Notes**

To initialize this node it is required to provide a bounding box of an object to track.
This is achieved by initializing one of the toolkit's 2D object detectors (YOLOv3) and running object detection once on the input.
Afterwards, **the detected bounding box that is closest to the center of the image** is used to initialize the tracker. 
Feel free to modify the node to initialize it in a different way that matches your use case.

### 2D Object Tracking ROS2 Nodes

For 2D object tracking, there two ROS2 nodes provided, one using Deep Sort and one using FairMOT which use either pretrained models, or custom trained models.
The predicted tracking annotations are split into two topics with detections and tracking IDs. Additionally, an annotated image is generated.

You can find the 2D object detection ROS2 node python scripts here: [Deep Sort node](./opendr_perception/object_tracking_2d_deep_sort_node.py) and [FairMOT node](./opendr_perception/object_tracking_2d_fair_mot_node.py)
where you can inspect the code and modify it as you wish to fit your needs.
The nodes makes use of the toolkit's [object tracking 2D - Deep Sort tool](../../../../src/opendr/perception/object_tracking_2d/deep_sort/object_tracking_2d_deep_sort_learner.py)
and [object tracking 2D - FairMOT tool](../../../../src/opendr/perception/object_tracking_2d/fair_mot/object_tracking_2d_fair_mot_learner.py)
whose documentation can be found here: [Deep Sort docs](../../../../docs/reference/object-tracking-2d-deep-sort.md), [FairMOT docs](../../../../docs/reference/object-tracking-2d-fair-mot.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start a 2D object tracking node:
   1. Deep Sort node
      ```shell
      ros2 run opendr_perception object_tracking_2d_deep_sort
      ```
      The following optional argument is available for the Deep Sort node:
      - `-n --model_name MODEL_NAME`: name of the trained model (default=`deep_sort`)
   2. FairMOT node
      ```shell
      ros2 run opendr_perception object_tracking_2d_fair_mot
      ```
      The following optional argument is available for the FairMOT node:
      - `-n --model_name MODEL_NAME`: name of the trained model (default=`fairmot_dla34`)

    The following optional arguments are available for both nodes:
   - `-h or --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_objects_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages, `None` to stop the node from publishing on this topic (default=`/opendr/objects`)
   - `-t or --tracking_id_topic TRACKING_ID_TOPIC`: topic name for tracking ID messages, `None` to stop the node from publishing on this topic (default=`/opendr/objects_tracking_id`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `-td --temp_dir TEMP_DIR`: path to a temporary directory with models (default=`temp`)

3. Default output topics:
   - Output images: `/opendr/image_objects_annotated`
   - Detection messages: `/opendr/objects`
   - Tracking ID messages: `/opendr/objects_tracking_id`

   For viewing the output, refer to the [notes above.](#notes)

**Notes**

An [image dataset node](#image-dataset-ros2-node) is also provided to be used along these nodes.
Make sure to change the default input topic of the tracking node if you are not using the USB cam node.

### Vision Based Panoptic Segmentation ROS2 Node

A ROS node for performing panoptic segmentation on a specified RGB image stream using the [EfficientPS](../../../../src/opendr/perception/panoptic_segmentation/README.md#efficientps-efficient-panoptic-segmentation) network.

You can find the vision based panoptic segmentation (EfficientPS) ROS node python script [here](./opendr_perception/panoptic_segmentation_efficient_ps_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [panoptic segmentation tool](../../../../src/opendr/perception/panoptic_segmentation/efficient_ps/efficient_ps_learner.py) whose documentation can be found [here](../../../../docs/reference/efficient-ps.md)
and additional information about EfficientPS [here](../../../../src/opendr/perception/panoptic_segmentation/README.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the panoptic segmentation node:

    ```shell
    ros2 run opendr_perception panoptic_segmentation_efficient_ps
    ```

    The following optional arguments are available:
   - `-h, --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC` : listen to RGB images on this topic (default=`/usb_cam/image_raw`)
   - `--checkpoint CHECKPOINT` : download pretrained models [cityscapes, kitti] or load from the provided path (default=`cityscapes`)
   - `-oh or --output_heatmap_topic OUTPUT_RGB_IMAGE_TOPIC`: publish the semantic and instance maps on this topic as `OUTPUT_HEATMAP_TOPIC/semantic` and `OUTPUT_HEATMAP_TOPIC/instance` (default=`/opendr/panoptic`)
   - `-ov or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: publish the panoptic segmentation map as an RGB image on `VISUALIZATION_TOPIC` or a more detailed overview if using the `--detailed_visualization` flag (default=`/opendr/panoptic/rgb_visualization`)
   - `--detailed_visualization`: generate a combined overview of the input RGB image and the semantic, instance, and panoptic segmentation maps and publish it on `OUTPUT_RGB_IMAGE_TOPIC` (default=deactivated)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)

3. Default output topics:
   - Output images: `/opendr/panoptic/semantic`, `/opendr/panoptic/instance`, `/opendr/panoptic/rgb_visualization`
   - Detection messages: `/opendr/panoptic/semantic`, `/opendr/panoptic/instance`

   For viewing the output, refer to the [notes above.](#notes)

### Semantic Segmentation ROS2 Node

You can find the semantic segmentation ROS2 node python script [here](./opendr_perception/semantic_segmentation_bisenet_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [semantic segmentation tool](../../../../src/opendr/perception/semantic_segmentation/bisenet/bisenet_learner.py) whose documentation can be found [here](../../../../docs/reference/semantic-segmentation.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the semantic segmentation node:

    ```shell
    ros2 run opendr_perception semantic_segmentation_bisenet
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/image_raw`)
   - `-o or --output_heatmap_topic OUTPUT_HEATMAP_TOPIC`: topic to which we are publishing the heatmap in the form of a ROS2 image containing class IDs, `None` to stop the node from publishing on this topic (default=`/opendr/heatmap`)
   - `-ov or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic to which we are publishing the heatmap image blended with the input image and a class legend for visualization purposes, `None` to stop the node from publishing on this topic (default=`/opendr/heatmap_visualization`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)

3. Default output topics:
   - Output images: `/opendr/heatmap`, `/opendr/heatmap_visualization`
   - Detection messages: `/opendr/heatmap`

   For viewing the output, refer to the [notes above.](#notes)

**Notes**

On the table below you can find the detectable classes and their corresponding IDs:

| Class  | Bicyclist | Building | Car | Column Pole | Fence | Pedestrian | Road | Sidewalk | Sign Symbol | Sky | Tree | Unknown |
|--------|-----------|----------|-----|-------------|-------|------------|------|----------|-------------|-----|------|---------|
| **ID** | 0         | 1        | 2   | 3           | 4     | 5          | 6    | 7        | 8           | 9   | 10   | 11      |

### Binary High Resolution ROS2 Node

You can find the binary high resolution ROS2 node python script [here](./opendr_perception/binary_high_resolution_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [binary high resolution tool](../../../../src/opendr/perception/binary_high_resolution/binary_high_resolution_learner.py) whose documentation can be found [here](../../../../docs/reference/binary_high_resolution.md).

#### Instructions for basic usage:

0. Before running this node it is required to train a model for a specific binary classification task. 
   Refer to the tool's [documentation](../../../../docs/reference/binary_high_resolution.md) for more information.
   To test the node out, run [train_eval_demo.py](../../../python/perception/binary_high_resolution/train_eval_demo.py)
   to download the test dataset provided and to train a test model. 
   You would then need to move the model folder in `opendr_ws_2` so the node can load it using the default `model_path` argument.

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the binary high resolution node:

    ```shell
    ros2 run opendr_perception binary_high_resolution
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/image_raw`)
   - `-o or --output_heatmap_topic OUTPUT_HEATMAP_TOPIC`: topic to which we are publishing the heatmap in the form of a ROS2 image containing class IDs, `None` to stop the node from publishing on this topic (default=`/opendr/binary_hr_heatmap`)
   - `-ov or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic to which we are publishing the heatmap image blended with the input image and a class legend for visualization purposes, `None` to stop the node from publishing on this topic (default=`/opendr/binary_hr_heatmap_visualization`)
   - `-m or --model_path MODEL_PATH`: path to the directory of the trained model (default=`test_model`)
   - `-a or --architecture ARCHITECTURE`: architecture used for the trained model, either `VGG_720p` or `VGG_1080p` (default=`VGG_720p`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)

3. Default output topics:
   - Output images: `/opendr/binary_hr_heatmap`, `/opendr/binary_hr_heatmap_visualization`
   - Detection messages: `/opendr/binary_hr_heatmap`

   For viewing the output, refer to the [notes above.](#notes)

### Image-based Facial Emotion Estimation ROS2 Node

You can find the image-based facial emotion estimation ROS2 node python script [here](./opendr_perception/facial_emotion_estimation_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's image-based facial emotion estimation tool which can be found [here](../../../../src/opendr/perception/facial_expression_recognition/image_based_facial_emotion_estimation/facial_emotion_learner.py)
whose documentation can be found [here](../../../../docs/reference/image_based_facial_emotion_estimation.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the image-based facial emotion estimation node:

    ```shell
    ros2 run opendr_perception facial_emotion_estimation
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_emotion_estimation_annotated`)
   - `-e or --output_emotions_topic OUTPUT_EMOTIONS_TOPIC`: topic to which we are publishing the facial emotion results, `None` to stop the node from publishing on this topic (default=`"/opendr/facial_emotion_estimation"`)
   - `-m or --output_emotions_description_topic OUTPUT_EMOTIONS_DESCRIPTION_TOPIC`: topic to which we are publishing the description of the estimated facial emotion, `None` to stop the node from publishing on this topic (default=`/opendr/facial_emotion_estimation_description`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)

3. Default output topics:
   - Output images: `/opendr/image_emotion_estimation_annotated`
   - Detection messages: `/opendr/facial_emotion_estimation`, `/opendr/facial_emotion_estimation_description`

   For viewing the output, refer to the [notes above.](#notes)

**Notes**

This node requires the detection of a face first. This is achieved by including of the toolkit's face detector and running face detection on the input.
Afterwards, the detected bounding box of the face is cropped and fed into the facial emotion estimator. 
Feel free to modify the node to detect faces in a different way that matches your use case.

### Landmark-based Facial Expression Recognition ROS2 Node

A ROS2 node for performing landmark-based facial expression recognition using a trained model on AFEW, CK+ or Oulu-CASIA datasets.
OpenDR does not include a pretrained model, so one should be provided by the user.
An alternative would be to use the [image-based facial expression estimation node](#image-based-facial-emotion-estimation-ros2-node) provided by the toolkit.

You can find the landmark-based facial expression recognition ROS2 node python script [here](./opendr_perception/landmark_based_facial_expression_recognition_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's landmark-based facial expression recognition tool which can be found [here](../../../../src/opendr/perception/facial_expression_recognition/landmark_based_facial_expression_recognition/progressive_spatio_temporal_bln_learner.py)
whose documentation can be found [here](../../../../docs/reference/landmark-based-facial-expression-recognition.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the landmark-based facial expression recognition node:

    ```shell
    ros2 run opendr_perception landmark_based_facial_expression_recognition
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/image_raw`)
   - `-o or --output_category_topic OUTPUT_CATEGORY_TOPIC`: topic to which we are publishing the recognized facial expression category info, `None` to stop the node from publishing on this topic (default=`"/opendr/landmark_expression_recognition"`)
   - `-d or --output_category_description_topic OUTPUT_CATEGORY_DESCRIPTION_TOPIC`: topic to which we are publishing the description of the recognized facial expression, `None` to stop the node from publishing on this topic (default=`/opendr/landmark_expression_recognition_description`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `--model`: architecture to use for facial expression recognition, options are `pstbln_ck+`, `pstbln_casia`, `pstbln_afew` (default=`pstbln_afew`)
   - `-s or --shape_predictor SHAPE_PREDICTOR`: shape predictor (landmark_extractor) to use (default=`./predictor_path`)

3. Default output topics:
   - Detection messages: `/opendr/landmark_expression_recognition`, `/opendr/landmark_expression_recognition_description`

   For viewing the output, refer to the [notes above.](#notes)

### Skeleton-based Human Action Recognition ROS2 Nodes

A ROS2 node for performing skeleton-based human action recognition is provided, one using either ST-GCN or PST-GCN models pretrained on NTU-RGBD-60 dataset.
Another ROS2 node for performing continual skeleton-based human action recognition is provided, using the CoSTGCN method. 
The human body poses of the image are first extracted by the lightweight OpenPose method which is implemented in the toolkit, and they are passed to the skeleton-based action recognition methods to be categorized.

You can find the skeleton-based human action recognition ROS2 node python script [here](./opendr_perception/skeleton_based_action_recognition_node.py) 
and the continual skeleton-based human action recognition ROS2 node python script [here](./opendr_perception/continual_skeleton_based_action_recognition_node.py) to inspect the code and modify it as you wish to fit your needs.
The latter makes use of the toolkit's skeleton-based human action recognition tool which can be found [here for ST-GCN](../../../../src/opendr/perception/skeleton_based_action_recognition/spatio_temporal_gcn_learner.py)
and [here for PST-GCN](../../../../src/opendr/perception/skeleton_based_action_recognition/progressive_spatio_temporal_gcn_learner.py) and the former makes use
of the toolkit's continual skeleton-based human action recognition tool which can be found [here](../../../../src/opendr/perception/skeleton_based_action_recognition/continual_stgcn_learner.py).
Their documentation can be found [here](../../../../docs/reference/skeleton-based-action-recognition.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the skeleton-based human action recognition node:
   1. Skeleton-based action recognition node
      ```shell
      ros2 run opendr_perception skeleton_based_action_recognition
      ```
      The following optional argument is available for the skeleton-based action recognition node:
      - `--model` MODEL: model to use, options are `stgcn` or `pstgcn`, (default=`stgcn`)
      - `-c or --output_category_topic OUTPUT_CATEGORY_TOPIC`: topic name for recognized action category, `None` to stop the node from publishing on this topic (default=`"/opendr/skeleton_recognized_action"`)
      - `-d or --output_category_description_topic OUTPUT_CATEGORY_DESCRIPTION_TOPIC`: topic name for description of the recognized action category, `None` to stop the node from publishing on this topic (default=`/opendr/skeleton_recognized_action_description`)

   2. Continual skeleton-based action recognition node
      ```shell
      ros2 run opendr_perception continual_skeleton_based_action_recognition
      ```
      The following optional argument is available for the continual skeleton-based action recognition node:
      - `--model` MODEL: model to use, options are `costgcn`, (default=`costgcn`)
      - `-c or --output_category_topic OUTPUT_CATEGORY_TOPIC`: topic name for recognized action category, `None` to stop the node from publishing on this topic (default=`"/opendr/continual_skeleton_recognized_action"`)
      - `-d or --output_category_description_topic OUTPUT_CATEGORY_DESCRIPTION_TOPIC`: topic name for description of the recognized action category, `None` to stop the node from publishing on this topic (default=`/opendr/continual_skeleton_recognized_action_description`)

    The following optional arguments are available for all nodes:
   - `-h or --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/usb_cam/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output pose-annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_pose_annotated`)
   - `-p or --pose_annotations_topic POSE_ANNOTATIONS_TOPIC`: topic name for pose annotations, `None` to stop the node from publishing on this topic (default=`/opendr/poses`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)

3. Default output topics:
   1. Skeleton-based action recognition node:
      - Detection messages: `/opendr/skeleton_based_action_recognition`, `/opendr/skeleton_based_action_recognition_description`, `/opendr/poses`
      - Output images: `/opendr/image_pose_annotated`
   2. Continual skeleton-based action recognition node:
      - Detection messages: `/opendr/continual_skeleton_recognized_action`, `/opendr/continual_skeleton_recognized_action_description`, `/opendr/poses`
      - Output images: `/opendr/image_pose_annotated`

      For viewing the output, refer to the [notes above.](#notes)

### Video Human Activity Recognition ROS2 Node

A ROS2 node for performing human activity recognition using either CoX3D or X3D models pretrained on Kinetics400.

You can find the video human activity recognition ROS2 node python script [here](./opendr_perception/video_activity_recognition_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's video human activity recognition tools which can be found [here for CoX3D](../../../../src/opendr/perception/activity_recognition/cox3d/cox3d_learner.py) and
[here for X3D](../../../../src/opendr/perception/activity_recognition/x3d/x3d_learner.py) whose documentation can be found [here](../../../../docs/reference/activity-recognition.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the video human activity recognition node:

    ```shell
    ros2 run opendr_perception video_activity_recognition
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/image_raw`)
   - `-o or --output_category_topic OUTPUT_CATEGORY_TOPIC`: topic to which we are publishing the recognized activity, `None` to stop the node from publishing on this topic (default=`"/opendr/human_activity_recognition"`)
   - `-od or --output_category_description_topic OUTPUT_CATEGORY_DESCRIPTION_TOPIC`: topic to which we are publishing the ID of the recognized action, `None` to stop the node from publishing on this topic (default=`/opendr/human_activity_recognition_description`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--model`: architecture to use for human activity recognition, options are `cox3d-s`, `cox3d-m`, `cox3d-l`, `x3d-xs`, `x3d-s`, `x3d-m`, or `x3d-l` (default=`cox3d-m`)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)

3. Default output topics:
   - Detection messages: `/opendr/human_activity_recognition`, `/opendr/human_activity_recognition_description`

   For viewing the output, refer to the [notes above.](#notes)

**Notes**

You can find the corresponding IDs regarding activity recognition [here](https://github.com/opendr-eu/opendr/blob/master/src/opendr/perception/activity_recognition/datasets/kinetics400_classes.csv).

### RGB Gesture Recognition ROS2 Node

For gesture recognition, the ROS2 [node](./opendr_perception/gesture_recognition_node.py) is based on the gesture recognition learner defined [here](../../../../src/opendr/perception/gesture_recognition/gesture_recognition_learner.py), and the documentation of the learner can be found [here](../../../../docs/reference/gesture-recognition-learner.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. Start the gesture recognition node:
   ```shell
   ros2 run opendr_perception gesture_recognition
   ```
   The following arguments are available:
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/usb_cam/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image (default=`/opendr/image_gesture_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages (default=`/opendr/gestures`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: Device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `--threshold THRESHOLD`: Confidence threshold for predictions (default=0.5)
   - `--model MODEL`: Config file name of the model that will be used (default=`plus_m_1.5x_416)`

3. Default output topics:
   - Output images: `/opendr/image_gesture_annotated`
   - Detection messages: `/opendr/gestures`

## RGB + Infrared input

### 2D Object Detection GEM ROS2 Node

You can find the object detection 2D GEM ROS2 node python script [here](./opendr_perception/object_detection_2d_gem_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [object detection 2D GEM tool](../../../../src/opendr/perception/object_detection_2d/gem/gem_learner.py)
whose documentation can be found [here](../../../../docs/reference/gem.md).

#### Instructions for basic usage:

1. First one needs to find points in the color and infrared images that correspond, in order to find the homography matrix that allows to correct for the difference in perspective between the infrared and the RGB camera.
   These points can be selected using a [utility tool](../../../../src/opendr/perception/object_detection_2d/utils/get_color_infra_alignment.py) that is provided in the toolkit.

2. Pass the points you have found as *pts_color* and *pts_infra* arguments to the [ROS2 GEM node](./opendr_perception/object_detection_2d_gem.py).

3. Start the node responsible for publishing images. If you have a RealSense camera, then you can use the corresponding node (assuming you have installed [realsense2_camera](http://wiki.ros.org/realsense2_camera)):

   ```shell
   roslaunch realsense2_camera rs_camera.launch enable_color:=true enable_infra:=true enable_depth:=false enable_sync:=true infra_width:=640 infra_height:=480
   ```

4. You are then ready to start the object detection 2d GEM node:

    ```shell
    ros2 run opendr_perception object_detection_2d_gem
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-ic or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/camera/color/image_raw`)
   - `-ii or --input_infra_image_topic INPUT_INFRA_IMAGE_TOPIC`: topic name for input infrared image (default=`/camera/infra/image_raw`)
   - `-oc or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/rgb_image_objects_annotated`)
   - `-oi or --output_infra_image_topic OUTPUT_INFRA_IMAGE_TOPIC`: topic name for output annotated infrared image, `None` to stop the node from publishing on this topic (default=`/opendr/infra_image_objects_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages, `None` to stop the node from publishing on this topic (default=`/opendr/objects`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)

5. Default output topics:
   - Output RGB images: `/opendr/rgb_image_objects_annotated`
   - Output infrared images: `/opendr/infra_image_objects_annotated`
   - Detection messages: `/opendr/objects`

   For viewing the output, refer to the [notes above.](#notes)

----
## RGBD input

### RGBD Hand Gesture Recognition ROS2 Node
A ROS2 node for performing hand gesture recognition using a MobileNetv2 model trained on HANDS dataset.
The node has been tested with Kinectv2 for depth data acquisition with the following drivers: https://github.com/OpenKinect/libfreenect2 and https://github.com/code-iai/iai_kinect2.

You can find the RGBD hand gesture recognition ROS2 node python script [here](./opendr_perception/rgbd_hand_gesture_recognition_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [hand gesture recognition tool](../../../../src/opendr/perception/multimodal_human_centric/rgbd_hand_gesture_learner/rgbd_hand_gesture_learner.py)
whose documentation can be found [here](../../../../docs/reference/rgbd-hand-gesture-learner.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images from an RGBD camera. Remember to modify the input topics using the arguments in step 2 if needed.

2. You are then ready to start the hand gesture recognition node:
    ```shell
    ros2 run opendr_perception rgbd_hand_gesture_recognition
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-ic or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/kinect2/qhd/image_color_rect`)
   - `-id or --input_depth_image_topic INPUT_DEPTH_IMAGE_TOPIC`: topic name for input depth image (default=`/kinect2/qhd/image_depth_rect`)
   - `-o or --output_gestures_topic OUTPUT_GESTURES_TOPIC`: topic name for predicted gesture class (default=`/opendr/gestures`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)

3. Default output topics:
   - Detection messages:`/opendr/gestures`

   For viewing the output, refer to the [notes above.](#notes)

----
## RGB + Audio input

### Audiovisual Emotion Recognition ROS2 Node

You can find the audiovisual emotion recognition ROS2 node python script [here](./opendr_perception/audiovisual_emotion_recognition_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [audiovisual emotion recognition tool](../../../../src/opendr/perception/multimodal_human_centric/audiovisual_emotion_learner/avlearner.py),
whose documentation can be found [here](../../../../docs/reference/audiovisual-emotion-recognition-learner.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).
2. Start the node responsible for publishing audio. If you have an audio capture device, then you can use the `audio_capture_node` as explained in the [prerequisites above](#prerequisites).
3. You are then ready to start the audiovisual emotion recognition node

    ```shell
    ros2 run opendr_perception audiovisual_emotion_recognition
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-iv or --input_video_topic INPUT_VIDEO_TOPIC`: topic name for input video, expects detected face of size 224x224 (default=`/image_raw`)
   - `-ia or --input_audio_topic INPUT_AUDIO_TOPIC`: topic name for input audio (default=`/audio`)
   - `-o or --output_emotions_topic OUTPUT_EMOTIONS_TOPIC`: topic to which we are publishing the predicted emotion (default=`/opendr/audiovisual_emotion`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--buffer_size BUFFER_SIZE`: length of audio and video in seconds, (default=`3.6`)
   - `--model_path MODEL_PATH`: if given, the pretrained model will be loaded from the specified local path, otherwise it will be downloaded from an OpenDR FTP server

4. Default output topics:
   - Detection messages: `/opendr/audiovisual_emotion`

   For viewing the output, refer to the [notes above.](#notes)

----
## Audio input

### Speech Command Recognition ROS2 Node

A ROS2 node for recognizing speech commands from an audio stream using MatchboxNet, EdgeSpeechNets or Quadratic SelfONN models, pretrained on the Google Speech Commands dataset.

You can find the speech command recognition ROS2 node python script [here](./opendr_perception/speech_command_recognition_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's speech command recognition tools:
[EdgeSpeechNets tool](../../../../src/opendr/perception/speech_recognition/edgespeechnets/edgespeechnets_learner.py), [MatchboxNet tool](../../../../src/opendr/perception/speech_recognition/matchboxnet/matchboxnet_learner.py), [Quadratic SelfONN tool](../../../../src/opendr/perception/speech_recognition/quadraticselfonn/quadraticselfonn_learner.py)
whose documentation can be found here:
[EdgeSpeechNet docs](../../../../docs/reference/edgespeechnets.md), [MatchboxNet docs](../../../../docs/reference/matchboxnet.md), [Quadratic SelfONN docs](../../../../docs/reference/quadratic-selfonn.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing audio. If you have an audio capture device, then you can use the `audio_capture_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the speech command recognition node

    ```shell
    ros2 run opendr_perception speech_command_recognition
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-i or --input_audio_topic INPUT_AUDIO_TOPIC`: topic name for input audio (default=`/audio`)
   - `-o or --output_speech_command_topic OUTPUT_SPEECH_COMMAND_TOPIC`: topic name for speech command output (default=`/opendr/speech_recognition`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--buffer_size BUFFER_SIZE`: set the size of the audio buffer (expected command duration) in seconds (default=`1.5`)
   - `--model MODEL`: the model to use, choices are `matchboxnet`, `edgespeechnets` or `quad_selfonn` (default=`matchboxnet`)
   - `--model_path MODEL_PATH`: if given, the pretrained model will be loaded from the specified local path, otherwise it will be downloaded from an OpenDR FTP server

3. Default output topics:
   - Detection messages, class id and confidence: `/opendr/speech_recognition`

   For viewing the output, refer to the [notes above.](#notes)

**Notes**

EdgeSpeechNets currently does not have a pretrained model available for download, only local files may be used.

### Speech Transcription ROS2 Node

A ROS2 node for speech transcription from an audio stream using Whisper or Vosk.

You can find the speech transcription ROS node python script [here](./opendr_perception/speech_transcription_node.py) to inspect the code and modify it as you wish to fit your needs.

The node makes use of the toolkit's speech transcription tools:
[Whipser tool](../../../../src/opendr/perception/speech_transcription/whisper/whisper_learner.py), [Vosk tool](../../../../src/opendr/perception/speech_transcription/vosk/vosk_learner.py) whose documentation can be found here:
[Whisper docs](../../../../docs/reference/speech-transcription-whisper.md), [Vosk docs](../../../../docs/reference/speech-transcription-vosk.md).


#### Instruction for basic usage:

1. Start the node responsible for publishing audio. The ROS2 node only work with audio data in WAVE format. If you have an audio capture device, then you can use the `audio_capture_node` as explained in the [prerequisites above](#prerequisites).
    ```shell
    ros2 launch audio_capture capture_wave.launch.xml
    ```

2. You are then ready to start the speech transcription node

    ```shell
    ros2 run opendr_perception speech_transcription --verbose True
    ```
    ```shell
    ros2 run opendr_perception speech_transcription --backbone whisper --model_name tiny.en --verbose True
    ```
    The following optional arguments are available (More in the source code):
   - `-h or --help`: show a help message and exit
   - `-i or --input_audio_topic INPUT_AUDIO_TOPIC`: topic name for input audio (default=`/audio/audio`)
   - `-o or --output_speech_transcription_topic OUTPUT_TRANSCRIPTION_TOPIC`: topic name for speech transcription output (default=`/opendr/speech_transcription`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--backbone {vosk,whisper}`: Backbone model for speech transcription
   - `--model_name MODEL_NAME`: Specific model name for each backbone. Example: 'tiny', 'tiny.en', 'base', 'base.en' for Whisper, 'vosk-model-small-en-us-0.15' for Vosk (default=`None`) 
   - `--model_path MODEL_PATH`: Path to downloaded model files (default=`None`) 
   - `--language LANGUAGE`: Whisper uses the language parameter to avoid language dectection. Vosk uses the langauge paremeter to select a specific model. Example: 'en' for Whisper, 'en-us' for Vosk (default=`en-us`). Check the available language codes for Whisper at [Whipser repository](https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/tokenizer.py#L10). Check the available language code for Vosk from the Vosk model name at [Vosk website](https://alphacephei.com/vosk/models).
   - `--verbose VERBOSE`: Display transcription (default=`False`) 

3. Default output topics:
   - Speech transcription: `/opendr/speech_transcription`

   For viewing the output, refer to the [notes above.](#notes)

----
## Point cloud input

### 3D Object Detection Voxel ROS2 Node

A ROS2 node for performing 3D object detection Voxel using PointPillars or TANet methods with either pretrained models on KITTI dataset, or custom trained models.

You can find the 3D object detection Voxel ROS2 node python script [here](./opendr_perception/object_detection_3d_voxel_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [3D object detection Voxel tool](../../../../src/opendr/perception/object_detection_3d/voxel_object_detection_3d/voxel_object_detection_3d_learner.py)
whose documentation can be found [here](../../../../docs/reference/voxel-object-detection-3d.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing point clouds. OpenDR provides a [point cloud dataset node](#point-cloud-dataset-ros2-node) for convenience.

2. You are then ready to start the 3D object detection node:

    ```shell
    ros2 run opendr_perception object_detection_3d_voxel
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-i or --input_point_cloud_topic INPUT_POINT_CLOUD_TOPIC`: point cloud topic provided by either a point_cloud_dataset_node or any other 3D point cloud node (default=`/opendr/dataset_point_cloud`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages (default=`/opendr/objects3d`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `-n or --model_name MODEL_NAME`: name of the trained model (default=`tanet_car_xyres_16`)
   - `-c or --model_config_path MODEL_CONFIG_PATH`: path to a model .proto config (default=`../../src/opendr/perception/object_detection3d/voxel_object_detection_3d/second_detector/configs/tanet/car/xyres_16.proto`)

3. Default output topics:
   - Detection messages: `/opendr/objects3d`

   For viewing the output, refer to the [notes above.](#notes)

### 3D Object Tracking AB3DMOT ROS2 Node

A ROS2 node for performing 3D object tracking using AB3DMOT stateless method.
This is a detection-based method, and therefore the 3D object detector is needed to provide detections, which then will be used to make associations and generate tracking ids.
The predicted tracking annotations are split into two topics with detections and tracking IDs.

You can find the 3D object tracking AB3DMOT ROS2 node python script [here](./opendr_perception/object_tracking_3d_ab3dmot_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [3D object tracking AB3DMOT tool](../../../../src/opendr/perception/object_tracking_3d/ab3dmot/object_tracking_3d_ab3dmot_learner.py)
whose documentation can be found [here](../../../../docs/reference/object-tracking-3d-ab3dmot.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing point clouds. OpenDR provides a [point cloud dataset node](#point-cloud-dataset-ros2-node) for convenience.

2. You are then ready to start the 3D object tracking node:

    ```shell
    ros2 run opendr_perception object_tracking_3d_ab3dmot
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-i or --input_point_cloud_topic INPUT_POINT_CLOUD_TOPIC`: point cloud topic provided by either a point_cloud_dataset_node or any other 3D point cloud node (default=`/opendr/dataset_point_cloud`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages, `None` to stop the node from publishing on this topic (default=`/opendr/objects3d`)
   - `-t or --tracking3d_id_topic TRACKING3D_ID_TOPIC`: topic name for output tracking IDs with the same element count as in detection topic, `None` to stop the node from publishing on this topic (default=`/opendr/objects_tracking_id`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `-dn or --detector_model_name DETECTOR_MODEL_NAME`: name of the trained model (default=`tanet_car_xyres_16`)
   - `-dc or --detector_model_config_path DETECTOR_MODEL_CONFIG_PATH`: path to a model .proto config (default=`../../src/opendr/perception/object_detection3d/voxel_object_detection_3d/second_detector/configs/tanet/car/xyres_16.proto`)

3. Default output topics:
   - Detection messages: `/opendr/objects3d`
   - Tracking ID messages: `/opendr/objects_tracking_id`

   For viewing the output, refer to the [notes above.](#notes)


### LiDAR Based Panoptic Segmentation ROS2 Node
A ROS node for performing panoptic segmentation on a specified pointcloud stream using the [EfficientLPS](../../../../src/opendr/perception/panoptic_segmentation/README.md#efficientlps-efficient-lidar-panoptic-segmentation) network.

You can find the lidar based panoptic segmentation ROS node python script [here](./opendr_perception/panoptic_segmentation_efficient_lps_node.py). You can further also find the point cloud 2 publisher ROS node python script [here](./opendr_perception/point_cloud_2_publisher_node.py), and more explanation [here](#point-cloud-2-publisher-ros-node).You can inspect the codes and make changes as you wish to fit your needs.
The EfficientLPS node makes use of the toolkit's [panoptic segmentation tool](../../../../src/opendr/perception/panoptic_segmentation/efficient_lps/efficient_lps_learner.py) whose documentation can be found [here](../../../../docs/reference/efficient-lps.md)
and additional information about EfficientLPS [here](../../../../src/opendr/perception/panoptic_segmentation/README.md).

#### Instructions for basic usage:

1.  First one needs to download SemanticKITTI dataset into POINTCLOUD_LOCATION as it is described in the [Panoptic Segmentation Datasets](../../../../src/opendr/perception/panoptic_segmentation/datasets/README.md). Then, once the SPLIT type is specified (train, test or "valid", default "valid"), the point **Point Cloud 2 Publisher** can be started using the following line:

- ```shell
  ros2 run opendr_perception point_cloud_2_publisher -d POINTCLOUD_LOCATION -s SPLIT
  ```
2. After starting the **PointCloud2 Publisher**, one can start **EfficientLPS Node** using the following line:

- ```shell
  ros2 run opendr_perception panoptic_segmentation_efficient_lps /opendr/dataset_point_cloud2
  ```

  The following optional arguments are available:
   - `-h, --help`: show a help message and exit
   - `-i or --input_point_cloud_2_topic INPUT_POINTCLOUD2_TOPIC` : Point Cloud 2 topic provided by either a point_cloud_2_publisher_node or any other 3D Point Cloud 2 Node (default=`/opendr/dataset_point_cloud2`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `-c or --checkpoint CHECKPOINT` : download pretrained models [semantickitti] or load from the provided path (default=`semantickitti`)
   - `-o or --output_heatmap_pointcloud_topic OUTPUT_HEATMAP_POINTCLOUD_TOPIC`: publish the 3D heatmap pointcloud on `OUTPUT_HEATMAP_POINTCLOUD_TOPIC` (default=`/opendr/panoptic`)
 
3. Default output topics:
   - Detection messages: `/opendr/panoptic`


----
## Biosignal input

### Heart Anomaly Detection ROS2 Node

A ROS2 node for performing heart anomaly (atrial fibrillation) detection from ECG data using GRU or ANBOF models trained on AF dataset.

You can find the heart anomaly detection ROS2 node python script [here](./opendr_perception/heart_anomaly_detection_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's heart anomaly detection tools: [ANBOF tool](../../../../src/opendr/perception/heart_anomaly_detection/attention_neural_bag_of_feature/attention_neural_bag_of_feature_learner.py) and
[GRU tool](../../../../src/opendr/perception/heart_anomaly_detection/gated_recurrent_unit/gated_recurrent_unit_learner.py), whose documentation can be found here:
[ANBOF docs](../../../../docs/reference/attention-neural-bag-of-feature-learner.md) and [GRU docs](../../../../docs/reference/gated-recurrent-unit-learner.md).

#### Instructions for basic usage:

1. Start the node responsible for publishing ECG data.

2. You are then ready to start the heart anomaly detection node:

    ```shell
    ros2 run opendr_perception heart_anomaly_detection
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-i or --input_ecg_topic INPUT_ECG_TOPIC`: topic name for input ECG data (default=`/ecg/ecg`)
   - `-o or --output_heart_anomaly_topic OUTPUT_HEART_ANOMALY_TOPIC`: topic name for heart anomaly detection (default=`/opendr/heart_anomaly`)
   - `--performance_topic PERFORMANCE_TOPIC`: topic name for performance messages (default=`None`, disabled)
   - `--device DEVICE`: device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `--model MODEL`: the model to use, choices are `anbof` or `gru` (default=`anbof`)

3. Default output topics:
   - Detection messages: `/opendr/heart_anomaly`

   For viewing the output, refer to the [notes above.](#notes)

----
## Dataset ROS2 Nodes

The dataset nodes can be used to publish data from the disk, which is useful to test the functionality without the use of a sensor.
Dataset nodes use a provided `DatasetIterator` object that returns a `(Data, Target)` pair.
If the type of the `Data` object is correct, the node will transform it into a corresponding ROS2 message object and publish it to a desired topic.
The OpenDR toolkit currently provides two such nodes, an image dataset node and a point cloud dataset node.

### Image Dataset ROS2 Node

The image dataset node downloads a `nano_MOT20` dataset from OpenDR's FTP server and uses it to publish data to the ROS2 topic,
which is intended to be used with the [2D object tracking nodes](#2d-object-tracking-ros2-nodes).

You can create an instance of this node with any `DatasetIterator` object that returns `(Image, Target)` as elements,
to use alongside other nodes and datasets.
You can inspect [the node](./opendr_perception/image_dataset_node.py) and modify it to your needs for other image datasets.

To get an image from a dataset on the disk, you can start a `image_dataset.py` node as:
```shell
ros2 run opendr_perception image_dataset
```
The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-o or --output_rgb_image_topic`: topic name to publish the data (default=`/opendr/dataset_image`)
   - `-f or --fps FPS`: data fps (default=`10`)
   - `-d or --dataset_path DATASET_PATH`: path to a dataset (default=`/MOT`)
   - `-ks or --mot20_subsets_path MOT20_SUBSETS_PATH`: path to MOT20 subsets (default=`../../src/opendr/perception/object_tracking_2d/datasets/splits/nano_mot20.train`)

### Point Cloud Dataset ROS2 Node

The point cloud dataset node downloads a `nano_KITTI` dataset from OpenDR's FTP server and uses it to publish data to the ROS2 topic,
which is intended to be used with the [3D object detection node](#3d-object-detection-voxel-ros2-node),
as well as the [3D object tracking node](#3d-object-tracking-ab3dmot-ros2-node).

You can create an instance of this node with any `DatasetIterator` object that returns `(PointCloud, Target)` as elements,
to use alongside other nodes and datasets.
You can inspect [the node](./opendr_perception/point_cloud_dataset_node.py) and modify it to your needs for other point cloud datasets.

To get a point cloud from a dataset on the disk, you can start a `point_cloud_dataset.py` node as:
```shell
ros2 run opendr_perception point_cloud_dataset
```
The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-o or --output_point_cloud_topic`: topic name to publish the data (default=`/opendr/dataset_point_cloud`)
   - `-f or --fps FPS`: data fps (default=`10`)
   - `-d or --dataset_path DATASET_PATH`: path to a dataset, if it does not exist, nano KITTI dataset will be downloaded there (default=`/KITTI/opendr_nano_kitti`)
   - `-ks or --kitti_subsets_path KITTI_SUBSETS_PATH`: path to KITTI subsets, used only if a KITTI dataset is downloaded (default=`../../src/opendr/perception/object_detection_3d/datasets/nano_kitti_subsets`)

### Point Cloud 2 Publisher ROS2 Node

The point cloud 2 dataset publisher, publishes point cloud 2 messages from pre-downloaded dataset SemanticKITTI. It is currently being used by the ROS node [LiDAR Based Panoptic Segmentation ROS Node](#lidar-based-panoptic-segmentation-ros-node).

You can create an instance of this node with any `DatasetIterator` object that returns `(PointCloud, Target)` as elements,
to use alongside other nodes and datasets.
You can inspect [the node](./opendr_perception/point_cloud_2_publisher_node.py) and modify it to your needs for other point cloud datasets.

To get a point cloud from a dataset on the disk, you can start a `point_cloud_2_publisher_node.py` node as:
```shell
ros2 run opendr_perception point_cloud_2_publisher
```
The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-d or --dataset_path DATASET_PATH`: path of the SemanticKITTI dataset to publish the point cloud 2 message (default=`./datasets/semantickitti`)
   - `-s or --split SPLIT`: split of the dataset to use, only (train, valid, test) are available (default=`valid`)
   - `-o or --output_point_cloud_2_topic OUTPUT_POINT_CLOUD_2_TOPIC`: topic name to publish the data (default=`/opendr/dataset_point_cloud2`)
   - `-t or --test_data`: Add this argument if you want to only test this node with the test data available in our server

----
## Utility ROS2 Nodes

### Performance ROS2 Node

The performance node is used to subscribe to the optional performance topic of a running node and log its performance in terms of the time it
took to process a single input and produce output and in terms of frames per second. It uses a modifiable rolling window to calculate the average FPS.

You can inspect [the node](./opendr_perception/performance_node.py) and modify it to your needs.

#### Instructions for basic usage:

1. Start the node you want to benchmark as usual but also set the optional argument `--performance_topic` to, for example, `/opendr/performance`
2. Start the performance node:
    ```shell
    ros2 run opendr_perception performance
    ```
    The following optional arguments are available:
   - `-h or --help`: show a help message and exit
   - `-i or --input_performance_topic INPUT_PERFORMANCE_TOPIC`: topic name for input performance data (default=`/opendr/performance`)
   - `-w or --window WINDOW`: the window to use in number of frames to calculate the running average FPS (default=`20`)

Note that the `input_performance_topic` of the performance node must match the `performance_topic` of the running node.
Also note that the running node should properly get input and produce output to publish performance messages for the performance node to use.

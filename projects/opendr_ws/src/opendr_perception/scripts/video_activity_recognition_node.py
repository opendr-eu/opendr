#!/usr/bin/env python3
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

import argparse
import rospy
import torch
import torchvision
import cv2
from pathlib import Path
from std_msgs.msg import String
from vision_msgs.msg import ObjectHypothesis
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from opendr.engine.data import Video, Image
from opendr.perception.activity_recognition import CLASSES as KINETICS400_CLASSES
from opendr.perception.activity_recognition import CoX3DLearner
from opendr.perception.activity_recognition import X3DLearner


class HumanActivityRecognitionNode:
    def __init__(
        self,
        input_rgb_image_topic="/usb_cam/image_raw",
        output_category_topic="/opendr/human_activity_recognition",
        output_category_description_topic="/opendr/human_activity_recognition_description",
        device="cuda",
        model="cox3d-m",
    ):
        """
        Creates a ROS Node for video-based human activity recognition.
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_category_topic: Topic to which we are publishing the recognized activity
        (if None, we are not publishing the info)
        :type output_category_topic: str
        :param output_category_description_topic: Topic to which we are publishing the ID of the recognized action
         (if None, we are not publishing the ID)
        :type output_category_description_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param model:  Architecture to use for human activity recognition.
         (Options: 'cox3d-s', 'cox3d-m', 'cox3d-l', 'x3d-xs', 'x3d-s', 'x3d-m', 'x3d-l')
        :type model: str
        """

        assert model in {
            "cox3d-s",
            "cox3d-m",
            "cox3d-l",
            "x3d-xs",
            "x3d-s",
            "x3d-m",
            "x3d-l",
        }
        model_name, model_size = model.split("-")
        Learner = {"cox3d": CoX3DLearner, "x3d": X3DLearner}[model_name]

        # Initialize the human activity recognition
        self.learner = Learner(device=device, backbone=model_size)
        self.learner.download(path="model_weights", model_names={model_size})
        self.learner.load(Path("model_weights") / f"x3d_{model_size}.pyth")

        # Set up preprocessing
        if model_name == "cox3d":
            self.preprocess = _image_preprocess(
                image_size=self.learner.model_hparams["image_size"]
            )
        else:  # == x3d
            self.preprocess = _video_preprocess(
                image_size=self.learner.model_hparams["image_size"],
                window_size=self.learner.model_hparams["frames_per_clip"],
            )

        # Set up ROS topics and bridge
        self.input_rgb_image_topic = input_rgb_image_topic
        self.hypothesis_publisher = (
            rospy.Publisher(output_category_topic, ObjectHypothesis, queue_size=1)
            if output_category_topic
            else None
        )
        self.string_publisher = (
            rospy.Publisher(output_category_description_topic, String, queue_size=1)
            if output_category_description_topic
            else None
        )

        self.bridge = ROSBridge()

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node("opendr_human_activity_recognition_node", anonymous=True)
        rospy.Subscriber(
            self.input_rgb_image_topic,
            ROS_Image,
            self.callback,
            queue_size=1,
            buff_size=10000000,
        )
        rospy.loginfo("Human activity recognition node started.")
        rospy.spin()

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """
        image = self.bridge.from_ros_image(data, encoding="rgb8")
        if image is None:
            return

        x = self.preprocess(image.convert("channels_first", "rgb"))

        result = self.learner.infer(x)
        assert len(result) == 1
        category = result[0]
        category.confidence = float(category.confidence.max())  # Confidence for predicted class
        category.description = KINETICS400_CLASSES[category.data]  # Class name

        if self.hypothesis_publisher is not None:
            self.hypothesis_publisher.publish(self.bridge.to_ros_category(category))

        if self.string_publisher is not None:
            self.string_publisher.publish(
                self.bridge.to_ros_category_description(category)
            )


def _resize(image, size=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    if h > w:
        # calculate the ratio of the width and construct the
        # dimensions
        r = size / float(w)
        dim = (size, int(h * r))
    else:
        # calculate the ratio of the height and construct the
        # dimensions
        r = size / float(h)
        dim = (int(w * r), size)

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def _image_preprocess(image_size: int):
    standardize = torchvision.transforms.Normalize(
        mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)
    )

    def wrapped(frame):
        nonlocal standardize
        frame = frame.transpose((1, 2, 0))  # C, H, W -> H, W, C
        frame = _resize(frame, size=image_size)
        frame = torch.tensor(frame).permute((2, 0, 1))  # H, W, C -> C, H, W
        frame = frame / 255.0  # [0, 255] -> [0.0, 1.0]
        frame = standardize(frame)
        return Image(frame, dtype=float)

    return wrapped


def _video_preprocess(image_size: int, window_size: int):
    frames = []

    standardize = torchvision.transforms.Normalize(
        mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)
    )

    def wrapped(frame):
        nonlocal frames, standardize
        frame = frame.transpose((1, 2, 0))  # C, H, W -> H, W, C
        frame = _resize(frame, size=image_size)
        frame = torch.tensor(frame).permute((2, 0, 1))  # H, W, C -> C, H, W
        frame = frame / 255.0  # [0, 255] -> [0.0, 1.0]
        frame = standardize(frame)
        if not frames:
            frames = [frame for _ in range(window_size)]
        else:
            frames.pop(0)
            frames.append(frame)
        vid = Video(torch.stack(frames, dim=1))
        return vid

    return wrapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rgb_image_topic", help="Topic name for input rgb image",
                        type=str, default="/usb_cam/image_raw")
    parser.add_argument("-o", "--output_category_topic", help="Topic to which we are publishing the recognized activity",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/human_activity_recognition")
    parser.add_argument("-od", "--output_category_description_topic",
                        help="Topic to which we are publishing the ID of the recognized action",
                        type=lambda value: value if value.lower() != "none" else None,
                        default="/opendr/human_activity_recognition_description")
    parser.add_argument("--device",  help='Device to use, either "cpu" or "cuda", defaults to "cuda"',
                        type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", help="Architecture to use for human activity recognition.",
                        type=str, default="cox3d-m",
                        choices=["cox3d-s", "cox3d-m", "cox3d-l", "x3d-xs", "x3d-s", "x3d-m", "x3d-l"])
    args = parser.parse_args()

    try:
        if args.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif args.device == "cuda":
            print("GPU not found. Using CPU instead.")
            device = "cpu"
        else:
            print("Using CPU.")
            device = "cpu"
    except:
        print("Using CPU.")
        device = "cpu"

    human_activity_recognition_node = HumanActivityRecognitionNode(
        input_rgb_image_topic=args.input_rgb_image_topic,
        output_category_topic=args.output_category_topic,
        output_category_description_topic=args.output_category_description_topic,
        device=device,
        model=args.model,
    )
    human_activity_recognition_node.listen()


if __name__ == "__main__":
    main()

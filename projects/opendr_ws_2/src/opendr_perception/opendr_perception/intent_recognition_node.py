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
import torch
from time import perf_counter

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from vision_msgs.msg import ObjectHypothesis
import os
from opendr_bridge import ROS2Bridge
from opendr.perception.multimodal_human_centric import IntentRecognitionLearner
from opendr_bridge.msg import OpenDRTranscription

LABELS = [
            'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize',
            'Agree', 'Taunt', 'Flaunt', 'Joke', 'Oppose', 'Comfort',
            'Care', 'Inform', 'Advise', 'Arrange', 'Introduce',
            'Leave', 'Prevent', 'Greet', 'Ask for help'
         ]


class IntentRecognitionNode(Node):

    def __init__(
            self,
            input_transcription_topic="/opendr/transcription",
            output_intent_topic="/opendr/intent",
            performance_topic=None,
            device="cuda",
            text_backbone="bert-base-uncased", cache_path='cache'):
        """
        Creates a ROS2 Node for intent recognition.
        :param input_transcription_topic: topic where input text is published
        :type input_transcription_topic: str
        :param output_intent_topic: Topic to which we are publishing the predicted intent
        :type output_intent_topic: str
        :param performance_topic: Topic to which we are publishing performance information (if None, no performance
        message is published)
        :type performance_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param text_backbone: the name of the text backbone to use
        :type model: str
        :param cache_path: cache path
        :type cache_path: str
        """
        super().__init__('intent_recognition_node')

        self.input_transcription_subscriber = self.create_subscription(
            OpenDRTranscription, input_transcription_topic, self.callback, 1)

        if output_intent_topic is not None:
            self.intent_publisher = self.create_publisher(
                ObjectHypothesis, output_intent_topic, ObjectHypothesis, 1)
        else:
            self.intent_publisher = None

        if performance_topic is not None:
            self.performance_publisher = self.create_publisher(
                Float32, performance_topic, 1)
        else:
            self.performance_publisher = None

        self.bridge = ROS2Bridge()

        if text_backbone in ['bert-small', 'bert-mini', 'bert-tiny']:
            backbone = 'prajjwal1/'+text_backbone
        else:
            backbone = text_backbone

        # Initialize the learner
        self.learner = IntentRecognitionLearner(text_backbone=backbone, mode='language',
                                                device=device, cache_path=cache_path)
        if not os.path.exists('pretrained_models/{}.pth'.format(text_backbone)):
            self.learner.download('pretrained_models/')
        self.learner.load('pretrained_models/{}.pth'.format(text_backbone))

        self.last_phrase = ""

        self.get_logger().info("Intent recognition node initialized")

    def callback(self, data):
        """
        Callback that processes the input data and publishes to the corresponding topics.
        :param data: input message
        :type data: OpenDRTranscription
        """
        if self.performance_publisher:
            start_time = perf_counter()
        if self.last_phrase == data.incremental:
            # dummy check if new message is the same as previous
            # to deal with whisper's double messages in short phrases
            return
        self.last_phrase = data.incremental
        # Run intent recognition
        predictions = self.learner.infer({'text': data.incremental})
        for pred in predictions:
            print(LABELS[pred.data])
        if self.performance_publisher:
            end_time = perf_counter()
            fps = 1.0 / (end_time - start_time)  # NOQA
            fps_msg = Float32()
            fps_msg.data = fps
            self.performance_publisher.publish(fps_msg)

        if self.intent_publisher is not None:
            for prediction in predictions:
                self.intent_publisher.publish(
                    self.bridge.to_ros_category(prediction))


def main():
    rclpy.init(args=None)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_transcription_topic",
        help="Topic name for input transcription",
        type=str,
        default="/opendr/transcription")
    parser.add_argument(
        "-o",
        "--output_intent_topic",
        help="Topic name for output intent",
        type=lambda value: value if value.lower() != "none" else None,
        default="/opendr/intent")
    parser.add_argument(
        "--performance_topic",
        help="Topic name for performance messages, disabled (None) by default",
        type=str,
        default=None)
    parser.add_argument(
        "--device",
        help="Device to use (cpu, cuda)",
        type=str,
        default="cuda",
        choices=[
            "cuda",
            "cpu"])
    parser.add_argument(
        "--text_backbone",
        help="Text backbone that will be used",
        type=str,
        default="bert-base-uncased",
        choices=["bert-base-uncased", "albert-base-v2", "bert-small", "bert-mini", "bert-tiny"])

    parser.add_argument(
        "--cache_path",
        help="Text backbone that will be used",
        type=str,
        default="./cache/")
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
    except BaseException:
        print("Using CPU.")
        device = "cpu"

    intent_node = IntentRecognitionNode(
        device=device,
        text_backbone=args.text_backbone,
        input_transcription_topic=args.input_transcription_topic,
        output_intent_topic=args.output_intent_topic,
        performance_topic=args.performance_topic, cache_path=args.cache_path)
    intent_node.listen()


if __name__ == '__main__':
    main()

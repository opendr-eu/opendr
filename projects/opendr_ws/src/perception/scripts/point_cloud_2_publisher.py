import os
import rospy
import time
import argparse

from sensor_msgs.msg import PointCloud2 as ROS_PointCloud2
from opendr_bridge import ROSBridge

from opendr.engine.datasets import DatasetIterator
from opendr.perception.panoptic_segmentation import SemanticKittiDataset


class PointCloud2DatasetNode:
    def __init__(
        self,
        dataset: DatasetIterator,
        output_point_cloud_2_topic="/opendr/dataset_point_cloud2"
    ):
        """
        Creates a ROS Node for publishing dataset point clouds
        """

        # Initialize the face detector
        self.dataset = dataset
        # Initialize OpenDR ROSBridge object
        self.bridge = ROSBridge()

        if output_point_cloud_2_topic is not None:
            self.output_point_cloud_2_publisher = rospy.Publisher(
                output_point_cloud_2_topic, ROS_PointCloud2, queue_size=10
            )

    def start(self):
        i = 0

        while not rospy.is_shutdown():

            point_cloud = self.dataset[i % len(self.dataset)][0]

            rospy.loginfo("Publishing point_cloud_2 [" + str(i) + "]")
            message = self.bridge.to_ros_point_cloud2(
                point_cloud
            )
            self.output_point_cloud_2_publisher.publish(message)

            time.sleep(0.1)

            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # I have defined the default path in the place where I have my example data, this should be adjusted later on
    parser.add_argument('--dataset_path', type=str, default='/home/canakcia/datasets/semantickitti/test_data',
                        help='listen to RGB pointclouds on this topic')

    rospy.init_node('opendr_point_cloud_2_dataset')

    args = parser.parse_args()

    dataset = SemanticKittiDataset(path=os.path.join(args.dataset_path, "eval_data"), split="valid")

    dataset_node = PointCloud2DatasetNode(dataset)
    dataset_node.start()

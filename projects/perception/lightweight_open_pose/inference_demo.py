import cv2
import time
from perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from perception.pose_estimation.lightweight_open_pose.utilities import draw
import argparse
from os.path import join
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--accelerate", help="Enables acceleration flags (e.g., stride)", default=False,
                        action="store_true")
    args = parser.parse_args()

    onnx, device, accelerate = args.onnx, args.device, args.accelerate
    if accelerate:
        stride = True
        stages = 0
        half_precision = True
    else:
        stride = False
        stages = 2
        half_precision = False

    pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=stages,
                                                mobilenet_use_stride=stride, half_precision=half_precision)
    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("openpose_default")

    # Download one sample image
    pose_estimator.download(path=".", mode="test_data")
    image_path = join("temp", "dataset", "image", "000000000785.jpg")
    img = cv2.imread(image_path)

    if onnx:
        pose_estimator.optimize()

    poses = pose_estimator.infer(img)
    for pose in poses:
        draw(img, pose)
    cv2.imshow('Results', img)
    cv2.waitKey(0)



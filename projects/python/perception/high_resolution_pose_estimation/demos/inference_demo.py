
import cv2
from opendr.perception.pose_estimation import HighResolutionPoseEstimationLearner

from opendr.perception.pose_estimation import draw
from opendr.engine.data import Image
import argparse
from os.path import join


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--accelerate", help="Enables acceleration flags (e.g., stride)", default=False,
                        action="store_true")
    parser.add_argument("--height1", help="Base height of resizing in first inference", default=540)
    parser.add_argument("--height2", help="Base height of resizing in second inference", default=540)
    parser.add_argument("--hrdata",help="Select the image resolution for inference",default=1440)

    args = parser.parse_args()

    onnx, device, accelerate,base_height1,base_height2,hrdata = args.onnx, args.device, args.accelerate, args.height1, args.height2,args.hrdata

    if hrdata == 1440:
        image_file="000000000785_1440.jpg"
    elif hrdata == 1080:
        image_file="000000000785_1080.jpg"
    elif hrdata == 720:
        image_file="000000000785_720.jpg"
    else:
        raise Exception("The inference image resolution is not valid")

    if accelerate:
        stride = True
        stages = 0
        half_precision = True
    else:
        stride = False
        stages = 2
        half_precision = False

    pose_estimator = HighResolutionPoseEstimationLearner(device=device, num_refinement_stages=stages,
                                                mobilenet_use_stride=stride, half_precision=half_precision)
    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("openpose_default")

    # Download one sample image
    pose_estimator.download(path=".", mode="test_data")

    image_path = join("temp2", "dataset", "image", image_file)

    img = Image.open(image_path)

    if onnx:
        pose_estimator.optimize()

    poses = pose_estimator.infer(img,base_height1,base_height2)

    img_cv = img.opencv()
    for pose in poses:
        draw(img_cv, pose)
    cv2.imshow('Results', img_cv)
    cv2.waitKey(0)


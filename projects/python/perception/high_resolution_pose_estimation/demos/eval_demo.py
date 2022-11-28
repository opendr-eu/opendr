
from opendr.perception.pose_estimation import HighResolutionPoseEstimationLearner
import argparse
from os.path import join
from opendr.engine.datasets import ExternalDataset
import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--accelerate", help="Enables acceleration flags (e.g., stride)", default=False,
                        action="store_true")
    parser.add_argument("--height1", help="Base height of resizing in first inference", default=360)
    parser.add_argument("--height2", help="Base height of resizing in second inference", default=540)
    parser.add_argument("--hrdata", help="Select the image resolution for inference", default=1440)

    args = parser.parse_args()

    onnx, device, accelerate,base_height1,base_height2,hrdata = args.onnx, args.device, args.accelerate, args.height1, args.height2,args.hrdata


    if hrdata == 1440:
        data_file="data_1440"
    elif hrdata == 1080:
        data_file="data_1080"
    elif hrdata == 720:
        data_file="data_720"
    else:
        raise Exception("The inference image resolution is not valid")


    if accelerate:
        stride = True
        stages = 0
        half_precision = True
    else:
        stride = True
        stages = 2
        half_precision = True

    pose_estimator = HighResolutionPoseEstimationLearner(device=device, num_refinement_stages=stages,
                                                mobilenet_use_stride=stride,
                                                half_precision=half_precision)
    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("openpose_default")

    if onnx:
        pose_estimator.optimize()

    # Download a sample dataset
    pose_estimator.download(path=".", mode="test_data")

    eval_dataset = ExternalDataset(path=join("temp2", "dataset",data_file), dataset_type="COCO")

    t0=time.time()
    results_dict = pose_estimator.eval(eval_dataset,base_height1,base_height2, use_subset=False, verbose=True, silent=True,
                                       images_folder_name="image", annotations_filename="annotation.json")
    t1 = time.time()
    print("\n Evaluation time:  ", t1 - t0,"seconds")
    print("Evaluation results = ", results_dict)
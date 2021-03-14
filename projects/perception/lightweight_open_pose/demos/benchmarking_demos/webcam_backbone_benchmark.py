from perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from webcam_fps_test import webcamTester

onnx = True
device = "cuda"  # "cpu", "cuda"

pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=2, mobilenet_use_stride=False)
pose_estimator.download(path="trainedModel")

pose_estimator.load("trainedModel")
if onnx:
    pose_estimator.optimize()

print("Testing on device:", device)
webcamTester(pose_estimator, minimum_iterations=10, iteration_limit=1000, stdev_limit=2.5, verbose=True, webcam_index=1)

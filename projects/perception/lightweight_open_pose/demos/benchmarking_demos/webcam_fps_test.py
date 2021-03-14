import cv2
import time
from statistics import pstdev, mean
from tqdm import tqdm


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def webcamTester(pose_estimator, stdev_limit=1.0, iteration_limit=100, minimum_iterations=10,
                 verbose=False, webcam_index=0):
    """
    Benchmarking utility function using cv2 to grab images from a webcam and benchmark FPS of provided pose estimator.
    This method will iterate forward passes of the provided pose_estimator, measuring FPS, until the
    FPS standard deviation drops below the stdev_limit provided or until the iteration_limit.

    :param pose_estimator: Pose estimator learner object to use for inference
    :type pose_estimator: OpenDR Learner
    :param stdev_limit: Minimum standard deviation to achieve in measurements
    :type stdev_limit: float
    :param iteration_limit: Maximum number of iterations
    :type iteration_limit: int
    :param minimum_iterations: The minimum number of iterations to start measuring stdev
    :type minimum_iterations: int
    :param verbose: Whether to print messages
    :type verbose: bool
    :param webcam_index: Index of webcam to use
    :type webcam_index: int
    :return: Average FPS and standard deviation
    :rtype: tuple(float, float)
    """
    if verbose:
        print("Iterating until FPS stabilizes...")

    image_provider = VideoReader(webcam_index)
    fps_list = []
    it = 0

    time.sleep(0.1)  # Needed for tqdm bar to print correctly
    pbar = tqdm(total=iteration_limit)

    for img in image_provider:
        start_time = time.perf_counter()
        _ = pose_estimator.infer(img)
        end_time = time.perf_counter()

        frame_time = end_time - start_time
        fps_list.append(1.0 / frame_time)
        stdev = pstdev(fps_list)

        if it >= iteration_limit or (stdev < stdev_limit and len(fps_list) > minimum_iterations):
            if verbose:
                if it >= iteration_limit:
                    print("Reached iteration limit - stdev:", stdev)
                else:
                    print("FPS stabilized - stdev:", stdev)
            avg_fps = mean(fps_list)
            if verbose:
                print("Average FPS:", round(avg_fps, 2))
            pbar.close()
            return avg_fps, stdev
        it += 1
        pbar.update(1)

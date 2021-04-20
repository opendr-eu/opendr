from utils.robot_interface import PoseRobot
from utils.active import LightweightOpenPoseLearner
from utils.pose_controller import PoseController
from utils.visualization import Visualizer, fall_handler_fn
import argparse
from utils.webots import initialize_webots_setup
from cv2 import VideoWriter, VideoWriter_fourcc
from os.path import join

if __name__ == '__main__':
    opendr = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--opendr", help="Enables additional OpenDR improvements", default=False, action="store_true")
    parser.add_argument("--active", help="Enables active perception (experimental)", default=False, action="store_true")
    parser.add_argument("--video", help="Writes the output to a video file", default=False, action="store_true")
    parser.add_argument('--setup', help="Selects some of the 3 available setups [0, 1]", type=int,
                        choices=[0, 1, 2], default=0)
    parser.add_argument('--web', help="Enables web streaming on 0.0.0.0:5000", default=False, action="store_true")
    parser.add_argument('--local', help="Enables streaming on a local window", default=False, action="store_true")
    parser.add_argument('--nocollision', help="Disables collision avoidance", default=False, action="store_true")
    parser.add_argument('--platform', help="Used to switch between different control commands for different platforms",
                        default='webots', type=str)

    args = parser.parse_args()

    pose_robot = PoseRobot(robot=args.platform)

    if args.setup == 0:
        print("Evaluating on setup 0 (no initial view, sitting target) ")
        mode = 'sitting_near_center_rear'
    elif args.setup == 1:
        print("Evaluating on setup 2 ")
        mode = 'standing_far_other_offset_front'
    elif args.setup == 2:
        print("Evaluating on setup 3 ")
        mode = 'sitting_far_offset_front'
    else:
        pass

    if args.platform == 'webots':
        initialize_webots_setup(pose_robot, mode)

    output_path = mode + '_' + str(args.active) + '_' + str(args.opendr)

    if args.opendr:
        pose_estimator = LightweightOpenPoseLearner(device='cuda', num_refinement_stages=2,
                                                    mobilenet_use_stride=True, half_precision=True)
        infer_delay = 0.15  # delay calculated based on FPS on jetbot

    else:
        pose_estimator = LightweightOpenPoseLearner(device='cuda', num_refinement_stages=2,
                                                    mobilenet_use_stride=False, half_precision=False)
        infer_delay = 0.43  # delay calculated based on FPS on jetbot

    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("./openpose_default")
    fall_handler_fn_file = lambda imgs: fall_handler_fn(imgs, output_file=output_path)

    if args.video:
        if args.platform == 'jetbot':
            video_writer = VideoWriter(join("results", output_path + '.avi'), VideoWriter_fourcc(*'XVID'), 40.0,
                                       (800, 600))
        else:
            video_writer = VideoWriter(join("results", output_path + '.avi'), VideoWriter_fourcc(*'XVID'), 40.0,
                                       (1920, 1080))
        visualizer = Visualizer(video_writer=video_writer, stream=args.web, local_display=args.local)
    else:
        visualizer = Visualizer(stream=args.web, local_display=args.local)

    pose_controller = PoseController(pose_robot, pose_estimator, visualizer.visualization_handler, fall_handler_fn_file,
                                     args.active, infer_delay, disable_collision=args.nocollision)
    try:
        while True:
            pose_controller.rotate_to_detect()
            pose_controller.monitor_target()
    except Exception as e:
        print("Caught: ", e)
        if visualizer.video_writer:
            visualizer.video_writer.release()
        pose_robot.kill_switch()

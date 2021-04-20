# fall_detection_demo

This repository contains a demo developed on Webots to demonstrate how OpenDR functionality can be used to provide a naive fall detector.

Functionality provided:
- [x] Webots environment
- [x] TIAGo robot inteface for Webots
- [x] Fall detection based on relative height
- [x] PID-based control
- [x] Simulate jetson nano performance
- [x] OpenCV stream for debugging
- [x] Flask web-based streaming
- [x] Video recording support
- [x] Collision checking in forward movement
- [x] Switching setups from cmd arguments
- [x] Jetbot interface and prelim tests
- [ ] Test and calibrate on real robot


Nice to have: 
- [ ] Active collision avoidance (too heavy to run on jetbot)
- [ ] Avoid resizing operations on jetson to exploit its ISP (minor benefits are expected)
- [ ] Fall detection based on pose and its orientation (probably height-based detection already works well for demonstration)


## Instructions
1. Download the OpenDR toolkit and set its path to `evaluate.sh` and `jetbot.sh`.
2. Install the additional requirements: `pip3 install -r requirements.txt`. For the JetBot please use the requirements_jetson.txt file.
3. You should be are ready to go.

To run the evaluation on your computer using Webots, please start the world contained in the project directory `simulation_pose`. Then you can run different evaluation by running `evaluate.sh`. You can switch between different setups (three supported) using the `--setup` flag, you can record the simulation output in a file using the `--video` flag, you can enable the OpenDR acceleration mode using the `--opendr` flag, you can turn on the active perception mode (experimental) using the `--active` flag, while you can switch off the basic RGB mono obstacle avoidance using the `--nocollision` flag. The implementation also exposes a flask-based web interface for monitoring, which can be enabled using the `--web` flag (see below).

To run the evaluation on jetbot you can simply use the `jetbot.sh`. Note that collision avoidance is not supported on jetbot due to the limited computational resources. Furthermore, you should make sure that the camera is facing upwards. Based on the actual configuration and calibration of you robot, you might need to finetune the default parameters set in [pose_controller.py](https://github.com/cidl-auth/fall_detection_demo/blob/main/utils/pose_controller.py#L87). The demo controller attempts to stop the jetbot motors if an exception is raised. However, if this fails, then you can use the `jetbot_kill.sh` for convenience. However, note that the response time of the script might be high, if the jetbot is loaded.

## Monitoring progress remotely
Setting the ``--web`` flag allows for monitoring the stream from the web browser (this is expected to be useful when the demo will be running on jetbot). You can access the interface at `localhost:5000`. The server also responds to connections from other machines.

![alt text](flask.png)

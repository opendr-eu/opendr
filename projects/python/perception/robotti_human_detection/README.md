# Simulation of human detection with Robotti

This folder contains an example how to perform human detection with the Robotti model in simulation.
The human detection is performed using YOLOV5x.

### Setup the environment

To run this simulation, you need to install:
- Webots R2023b or newer ([installation instructions](https://cyberbotics.com/doc/guide/installing-webots))
- `perception` module of the OpenDR toolkit ([installation instructions](https://github.com/opendr-eu/opendr/blob/master/docs/reference/installation.md)).
- Install additional libraries:
    ```sh
    pip install gym
    sudo apt install libopenblas0
    ```

Then, you need to compile some libraries needed by the simulation, by running the following commands in a terminal:
```sh
cd $OPENDR_HOME/projects/python/simulation/robotti_human_detection
export WEBOTS_HOME=/path/to/webots/installation
make
```



### Run the simulation

First open a terminal and navigate to this folder.

Start Webots and open the `webots/worlds/robotti_human_detection.wbt` world file:
```sh
export WEBOTS_HOME=/path/to/webots/installation
$WEBOTS_HOME/webots webots/worlds/robotti_human_detection.wbt
```

Execute this command in a different terminal to start the controller program of the Robotti:
```sh
export WEBOTS_HOME=/path/to/webots/installation
$WEBOTS_HOME/webots-controller $OPENDR_HOME/projects/python/simulation/robotti_human_detection/controllers/human_detection/human_detection.py
```

By default the YOLOV5x is run on CPU.
If you want to use CUDA device, you can simply run this command instead of the previous one:
```sh
$WEBOTS_HOME/webots-controller $OPENDR_HOME/projects/python/simulation/robotti_human_detection/controllers/human_detection/human_detection.py --cuda
```

The Robotti should now start to move and the camera images should appear in the robot window.

### Expected results

The Robotti should be able to detect the walking human when it comes close to the robot and stop.
Robotti will start moving again when the human move away.



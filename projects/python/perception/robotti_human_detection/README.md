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

Then, you need to compile some libraries needed by the simulation, by opening a terminal, navigating 
to this folder, i.e. `/opendr/projects/python/perception/robotti_human_detection`, and running:
```sh
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

In a different terminal, navigate to your OpenDR root and activate the toolkit environment:
```sh
source bin/activate.sh 
```
Then navigate to this folder and start the controller program of the Robotti:
```sh
export WEBOTS_HOME=/path/to/webots/installation
$WEBOTS_HOME/webots-controller webots/controllers/human_detection/human_detection.py
```
Finally, start the simulation by hitting the play button in Webots.

By default, the YOLOV5x is run on CPU.
If you want to use CUDA device, you can `--cuda` to the previous command:
```sh
$WEBOTS_HOME/webots-controller webots/controllers/human_detection/human_detection.py --cuda
```

The Robotti should now start to move and the display image with annotation of detected person should appear in the robot window.

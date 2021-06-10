# Map Simulator

Python script for generating odometry and measurement datasets to test SLAM Algorithms.

## Usage

In a ROS system with `roscore` running:

```commandline
python2.7 map_simulator.py [-p] [-s <search_paths>] [-h] -i <input_file> [-o <output_file>] [<param_name>:<param_value>]...
```

### Command Arguments
#### Positional Arguments
 * ***`-i <input_file>`***, ***`--input <input_file>`***: (String)<br/>
 Path to a JSON file containing the desired map, robot movement commands and parameters.

#### Optional Arguments
 * ***`-o <output_file>`***, ***`--output <output_file>`***: (Optional, String)<br/>
 Path and filename where the output ROSBag file is to be saved. If none is provided, only the visualization will be run, without generating a ROSBag.
 * ***`-p`***, ***`--preview`***: (Optional)<br/>
 Display a step-by-step imulation using MatPlotLib.
 * ***`-s <paths>`***, ***`--search_paths <paths>`***: (Optional, String) *[Default: ".:robots:maps"]*<br/>
 Search paths for input and include files, separated by colons :.
 * ***`-h`***, ***`--help`***: (Optional)<br/>
 Display usage help.
 
 #### Parameter Override
 Any parameter defined in the following section can be defined from the command line as well, and not just from the json files.
 Parameters entered through the command line will override any values previously defined in the json files.
 To enter a parameter, its name and value must be entered separated by a colon-equal sign and using spaces between different parameters.
 E.g.:<br/>
 ```
python2.7 map_simulator.py ... meas_per_move:=1 odom_frame:=odom_combined ...
```
For more information on the possible parameters, please review the next section.
 
 ## Parameters
 The JSON input files can define the following parameters.
 If a given parameter is not defined in any of the files (main input file or included sub-files), then the default value will be used.
 
 
 ### Include
  * **include**: *(list of strings) [Default: None]*<br/>
  Parameter files to be included during parsing for reusing parameters between experiments with a more convenient, reduced-typing, smarter approach.<br/>
  E.g.:<br/>
    ```json
    "include": [
        "common.json",
        "map.json",
        "movements.json"
    ]
    ```
 
The include procedure is done in the following manner:
  * ***recursively*** and in a ***depth-first*** order:<br/>
  If an included file contains *include* statement as well, the files listed are also included,<br/>
     E.g.: Input file contains:
     ```json
     "include": [
         "a.json",
         "b.json",
         "c.json"
    ]
     ```
     and file `a.json` contains
     ```json
     "include": [
         "d.json",
         "e.json"
    ]
     ```
     Then the include order will be: `a.json`, `d.json`, `e.json`, `b.json`, `c.json`.
  * with ***duplicate detection***:<br/>
  Parsed files are kept in a list and included only once, the first time (in depth-first order) it is included.
  * ***last-in***:<br/>
  The last definition of a parameter according to include order rules overrides any previous definitions.
  ***Note:*** Included files are always parsed before any parameters configured locally in the file.
 
 ### TF
 #### TF Frames
  * **odom_frame**: (String) *[Default: "odom"]*<br/>
  TF Frame for the odometry measurements.
  * **base_frame**: (String) *[Default: "base_link"]*<br/>
  TF Frame for the robot's base
  * **laser_frame**: (String) *[Default:"laser_link"]*<br/>
  TF Frame for the Laser Sensor
  
#### TF Transforms
 * **base_to_laser_tf**: (Array) *[Default: [[0.05, 0.0], [0]] ]*<br/>
 Transform between the base and laser frames in the format [[x, y], θ].
 
### Map
 * **obstacles**: (List of Objects) *[Default: [] (Empty Map)]*<br/>
 A world map is defined as a list of geometric obstacles.<br/>
   E.g.:
    ```json
    "obstacles": [
        {"type": "polygon", "vertices": [[-0.1, 1.0], [10.1, 1.0], [10.1, 1.1], [-0.1, 1.1]]},
        {"type": "circle", "center": [4.0, 0.0], "radius": 10},
        ...
    ]
    ```
    Each obstacle is defined as a dictionary with a mandatory *type* and additional parameters depending on the type of geometric construct used.
#### Obstacle Types
##### Polygon
Defined as a set of vertices connected by straight line segments.
###### Arguments
 * **vertices**: (Array)<br/>
 List of ordered pairs of 2D points [x, y]. The order is relevant, as the segments of polyline connecting them are constructed between each consecutive pair of points, and between the last and first points.<br/>
 * **opacity**: (float) *[Default: 1.0]*<br/>
 Determines the probability with which the obstacle reflects a beam. A value of 1.0 makes the object always reflective, while 0.0 is completely transparent (so why bother defining the polygon in the first place).
 
###### Example
```json
{"type": "polygon", "vertices": [[-0.1, 1.0], [10.1, 1.0], [10.1, 1.1], [-0.1, 1.1]], "opacity": 0.5}
```

### Robot Movements

* **start_pose**: (Array) *[Default: [[0.0, 0.0], 0.0] ]*<br/>
Position and Orientation [[x, y], θ] of the Robot's base frame at timestep *t = 0*.<br/>
E.g.: `"start_pose": [[0.5, 0.5], [0.0]]`.

* **initial_timestamp**: (Float | null) *[Default: null]*<br/>
Timestamp of the initial pose message in seconds using Python's `time.time()` representation.
If *null*, then the current system time will be used instead.<br/>
E.g.:
    * `"initial_timestamp": null`: Set initial stamp to current system time.
    * `"initial_timestamp": 1596820827.48`: Set initial stamp to *August 7th 2020, 19:20:27.480*.

* **move_time_interval**: (Float) *[Default: 1000.0]*<br/>
Unless otherwise specified, time in *ms* that each move command takes to execute.
Used to advance the time stamp in the odometry messages.<br/>
E.g.: `"move_time_interval": 1000.0`.

* **move_commands**: (List of Objects) *[Default: [] (No movement)]* Determines the position of the robot at every time step, and the total duration of the simulation.

    ```json
    "move_commands": [
        {"type": "pose", "params": [[1.5, 0.5], 0.0]},
        {"type": "pose", "params": [[2.5, 0.5], 0.0]},
        ...
    ]
    ```

    Each command is defined by a type and additional parameters.

#### Movement Types
##### Pose
Defines the exact position and orientation of the robot at a given time step.

###### Arguments
 * **params**: (Array) 2D Position and orientation [[x, y], θ]

###### Example
```json
{"type": "pose", "params": [[1.5, 0.5], 0.0]}
```

##### Pose
Defines the desired position and orientation of the robot at a given time step.
From them, an initial rotation, a translation and a final rotation will be computed with respect to the previous desired pose.

###### Arguments
 * **params**: (Array) 2D Position and orientation [[x, y], θ]

###### Example
```json
{"type": "odometry", "params": [[1.5, 0.5], 0.0]}
```


### Measurements

* **scan_topic**: (String) *[Default: "base_scan"]*<br/>
Name of the message topic where the measurements will be published.<br/>
E.g.: `"scan_topic": "base_scan"`.

* **meas_per_move**: (Int) *[Default: 5]*<br/>
Number of measurement scans to perform after each movement command.<br/>
E.g.: `"meas_per_move": 3`.

* **scan_time_interval**: (Float) *[Default: 50.0]*<br/>
Time in *ms* that each measurement scan takes to execute.
Used to advance the time stamp in the messages.<br/>
E.g.: `"scan_time_interval": 50.0`.
                        
* **max_range**: (Float) *[Default: 20.0]*<br/>
Maximum distance in *m* that a sensor can measure.
If a beam does not hit an obstacle, then the returned value will be this.<br/>
E.g.: `"max_range": 20.0`.

* **num_rays**: (Int) *[Default: 180]*<br/>
Number of measurements/beams/rays taken by the sensor in a single scan.
Equally spaced between *start_ray* and *end_ray*.<br/>
E.g.: `"num_rays": 1` for a single beam.
  
* **start_ray**: (Float) *[Default: -1.5707963268 (-π/2)]*<br/>
Start angle in radians for the scan rays.<br/>
E.g.: `"start_ray": -3.141592`

* **end_ray**: (Float) *[Default: 1.5707963268 (π/2)]*<br/>
End angle in radians for the scan rays.<br/>
E.g.: `"end_ray": 3.141592`

### Uncertainty

* **deterministic**: (Bool) *[Default: false]*<br/>
When true, then the ground truth positions and measurements will be displayed and saved to the ROSBag.
Otherwise, zero-mean gaussian noise will be added to the odometry and sensor data according to the following covariances. 

* **Pose_sigma**: (3x3 Matrix) *[Default: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.]] ]*<br/>
Covariance matrix defining the uncertainty in the robot's pose (Only used for *pose*-type movements).<br/>
    ```
          ⎛ σxx σxy σxθ ⎞
      Σ = ⎜ σyx σyy σyθ ⎟
          ⎝ σθx σθy σθθ ⎠
    ```
    E.g.:
    ```json
    "pose_sigma": [[0.01, 0.00, 0.0],
                   [0.00, 0.01, 0.0],
                   [0.00, 0.00, 0.1]],
    ```
* **odometry_alpha**: (4 Vector) *[Default: [0.0, 0.0, 0.0, 0.0] ]*<br/>
Weight parameters for odometry-based movements consisting of rotation,translation,rotation.<br/>
    ```
      α = ( α1 α2 α3 α4 )
    ```
    E.g.:
    ```json
    "odometry_alpha": [0.001, 0.01, 0.01, 0.001],
    ```
  
* **measurement_sigma**: (2x2 Matrix) *[Default: [[0.0, 0.0], [0.0, 0.0]]*<br/>
Covariance matrix defining the uncertainty in the robot's measurement bearings and ranges.<br/>
    ```
          ⎛ σφφ   σφz ⎞
      Σ = ⎜           ⎟
          ⎝ σzφ   σzz ⎠
    ```
    E.g.:
    ```json
    "measurement_sigma": [[0.010, 0.000],
                          [0.000, 0.002]]
    ```

### Visualization

* **render_move_pause**: (float) *[Default: 0.5]*<br/>
Time in seconds that the simulation will pause after displaying a movement.

* **render_sense_pause**: (float) *[Default: 0.35]*<br/>
Time in seconds that the simulation will pause after displaying a measurement.


---

JAB 2020
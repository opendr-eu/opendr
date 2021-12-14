# FMP SLAM Evaluation

This module includes helper nodes and scripts, useful for characterizing and evaluating the results of some SLAM algorithms.
# Nodes

## err_collector

This node collects the translational, rotational and total errors computed by another node and stores them in CSV files.

Errors during the full-SLAM (mapping and localization) phase of a run are stored in one file, while (if used) errors
measured during the Localization-Only phase are stored in a separate file. 

### Subscribed Topics

* **doLocOnly** ***(std_msg/Bool)***__:__<br/>
    When the Localization-Only phase is started and the doLocOnly message is issued, the node switches from saving to the
    mapping error file to the localization error file.

* **tra_err** ***(std_msg/Float64)***__:__<br/> Translational error of the SLAM corrected pose with respect to the ground truth pose.
* **rot_err** ***(std_msg/Float64)***__:__<br/> Rotational error of the SLAM corrected pose with respect to the ground truth pose.
* **tot_err** ***(std_msg/Float64)***__:__<br/> Total (translational + rotational) error of the SLAM corrected pose with respect to the ground truth pose.

### Parameters
* **~file_path** ***(str, default:*** __"~/Desktop/Experiments"__***)***__:__<br/>
    Directory where the files with the collected errors will be stored in a CSV format.
  
    Two files are constructed per run: a Mapping and a Localization-Only phase positional error files.
    The path to the files is constructed as follows:
    ```
    <file_path>/<file_prefix><suffix>
    ```
  
    where:
    
    * ***\<file_prefix\>*** is another parameter, explained in the following entry, and
    * ***\<suffix\>*** takes the value of either the *\<mapping\_suffix\>* or
    *<localization\_suffix>* parameter depending on the phase of the evaluation.

* **~file_prefix** ***(str, default:*** __"error_data"__***)***__:__<br/>
    Prefix for the file name of both the mapping error, and the localization-only error files.

* **~mapping_suffix** ***(str, default:*** __"\_map"__***)***__:__<br/>
    Suffix for the Mapping Phase error file.

* **~localization_suffix** ***(str, default:*** __"\_loc"__***)***__:__<br/>
    Suffix for the Localization-Only Phase error file.



## fmp_plot
Node for plotting the Full Map Posterior distribution and its properties as color maps. Depending on the parameters, it can
either save the plots to a directory, publish them to a topic, or both.

### Subscribed Topics
* **map_model** ***(gmapping/mapModel)***__:__<br/>
    Message specifying whether the SLAM algorithm is using the Reflection or Exponential Decay Rate map model.
    
    The used measurement likelihood functions and the resulting Map Posterior Distribution will depend on this parameter.
    
    The parametric, closed-form map posterior distribution will be either a Beta distribution (Beta(x; α, β)) for a Reflection Model,
  or a Gamma distribution (Gamma(x; α, β)) for Exponential Decay Rate model.
    
* **fmp_alpha** ***(gmapping/doubleMap)***__:__<br/>
    Map of the alpha parameter of the distribution (either Beta or Gamma according to the map_model setting).
* **fmp_beta** ***(gmapping/doubleMap)***__:__<br/>
    Map of the beta parameter of the distribution (either Beta or Gamma according to the map_model setting).

### Published Topics
If parameter ***~pub_image*** is set to True, then the node will publish the computed map distribution properties as images under the topics:


* **/\<pub_topic_prefix\>/<\topic\>** ***(sensor_msgs/Image)***<br/>

where:
* *\<pub_topic_prefix\>*: is a parameter determining the prefix to be used by all published topics.
* *\<topic\>*: are the topic names for the different distribution properties that can be published, namely:
    * *stats/mean* and *stats/var*: for the mean and variance of the map posterior.
    * *mlm*: for the raw, un-thresholded most likely map
    * *param/alpha* and *param/beta*: for publishing the distribution's parameters as images.

### Parameters

* **~img_stat** ***(bool, default:*** __False__***)***__:__<br/>
* **~img_mlm** ***(bool, default:*** __False__***)***__:__<br/>
* **~img_para** ***(bool, default:*** __False__***)***__:__<br/>
If set to true, generate the plots for the statistics (mean and variance), most-likely-map (raw, un-thresholded), and parameter (alpha and beta) respectively.


* **~pub_img** ***(bool, default:*** __False__***)***__:__<br/>
Publish the generated plots as images if set to true.
* **~pub_topic_prefix** ***(str, default:*** __"/fmp_img/"__***)***__:__<br/>
Prefix to be prepended to the published map plot topics.

* **~save_img** ***(bool, default:*** __False__***)***__:__<br/>
Save the generated plots as image files if set to true.
* **~resolution** ***(int, default:*** __300__***)***__:__<br/>
Resolution (in ppi) for the saved images.
* **~path_prefix** ***(str, default:*** __"exp"__***)***__:__<br/>
Prefix of the folder where the images are to be saved. Full folder name is constructed as <path_prefix>_<ts>, where <ts>
is the time of execution formatted as *"yymmdd_HHMMSS"*, unless a full path is explicitly set under parameter ***~save_dir***.
* **~save_dir** ***(str, default:*** __"~/Desktop/FMP_img/\<path\_prefix\>\_\<ts\>"__***)***__:__<br/>
Full path where the images are going to be stored.



## gt_mapping
Node for computing the reflection map by considering the odometry transformations and the sensor scans as noise-free without
having to use a full SLAM stack. Maps are published either when the node is stopping, or by command using a ***genMap*** message.

This node is designed to make it easier to compare the map generated by a SLAM algorithm with a Ground Truth, or pure odometry map by using the same size, resolution and origin point.

### Subscribed Topics
* **/map** ***(nav_msgs/OccupancyGrid)***__:__<br/>
Map generated by a SLAM algorithm. Used for determining the map's width, height, resolution and origin if not run as standalone node.

* **endOfSim** ***(std_msg/Bool)***__:__<br/>
Message signaling the end of a simulation. Used for stopping the node after all buffered scans have been processed and a map is published, in order to make it easier automate tests.

* **genMap** ***(std_msg/Bool)***__:__<br/>
Signal for manually forcing to publish a map with the data obtained so far, instead of waiting for the node to stop.

* **doLocOnly** ***(std_msg/Bool)***__:__<br/>
Signal that signifies that the SLAM algorithm will perform localization only, i.e. without updating its map, thus signaling
this node to also not update the map with scans received after this message is received.

* **/GT/base_scan** ***(sensor_msgs/LaserScan)***__:__<br/>
LiDAR sensor measurements used for localization and map generation.

### Published Topics
* **/GT/map** ***(nav_msgs/OccupancyGrid)***__:__<br/>
Map generated by considering the poses and measurements as noise-free.

### Parameters
* **~map_frame** ***(str, default:*** __"map"__***)***__:__<br/>
* **~ref_frame** ***(str, default:*** __"map"__***)***__:__<br/>
TF Frames for where to place the map's origin and the reference frame for the sensor respectively.

* **~max_scan_buffer_len** ***(int, default:*** __1000__***)***__:__<br/>
Maximum size of the buffer holding unprocessed scans, i.e. scans received before receiving a map message to determine the map's
resolution in case not set to standalone.

* **~occ_threshold** ***(float, default:*** __0.25__***)***__:__<br/>
Threshold value for establishing the value of each cell as either free or occupied. Only values in [0, 1] allowed.

* **~standalone** ***(bool, default:*** __False__***)***__:__<br/>
If set to true, then the node will not take the size, resolution or origin from another map message.

* **~resolution** ***(float, default:*** __0.05__***)***__:__<br/>
Grid cell size in meters. Only used if ***~standalone*** is set to true.




## occ_map_saver

Simple node for saving published Occupancy Maps to image files.

### Subscribed Topics
* **/map** ***(nav_msgs/OccupancyGrid)***__:__<br/>
Map to be saved to disc in *.png* format.

### Parameters
* **~path_prefix** ***(str, default:*** __"map"__***)***__:__<br/>
Name of the directory where the map images are going to be saved. Appended to the execution timestamp unless a full path is explicitly set in ***~save_dir***.
* **~file_prefix** ***(str, default:*** __"map"__***)***__:__<br/>
Name of the image files that are going to be saved. A sequence number will be appended to this prefix to construct the full filename.
* **~save_dir** ***(str, default:*** __"~/Desktop/OccMap/\<ts>\_\<path_prefix\>"__***)***__:__<br/>
Explicit path to the save directory.




## odom_pose

Node for taking the noisy poses, and outputting them under a different transform tree in order to visualize the pure odometry (i.e. uncorrected) movements.
Although useful, it is preferred to replicate the pure odometry transforms and scans directly within the ROSBag for better overall performance.

### Subscribed Topics
* **tf** ***(tf2_msgs/TFMessage)***__:__<br/>
Transform messages. Used to determine when a transform has been published between the configured coordinate frames, instead of
constantly polling for transforms.

### Published Topics
* **tf** ***(tf2_msgs/TFMessage)***__:__<br/>
Publishes the same transformations under a tree with prefix ***~frame_prefix*** and a static, identity transform between
the ***~map_frame*** and the ***~odom_frame*** as the pure-odometry transformations.

### Parameters
* **~map_frame** ***(str, default:*** __"map"__***)***__:__<br/>
* **~odom_frame** ***(str, default:*** __"odom"__***)***__:__<br/>
Name of the ROS TF Frames for the map and odometry coordinate frames respectively.
* **~frame_list** ***(str, default:*** __"[base_link, laser_link]"__***)***__:__<br/>
Comma-separated list of ROS TF Frames to filter the TF Messages with (along with the ***~odom_frame***), and to republish
under the new TF Tree with prefix ***~frame_prefix***.

* **~frame_prefix** ***(str, default:*** __"odo"__***)***__:__<br/>
Prefix to be added to the ROS TF Frames configured in the ***~frame_list*** (and ***~odom_frame***).
E.g. for an existing frame *base_link* a new one will be created called *odo/base_link* and the noisy, dynamic transforms will be replicated for it.




## pose_error_calc
Node for computing the translational, rotational and total error of a given pose with respect to the ground truth pose.

### Suscribed Topics
* **tf** ***(tf2_msgs/TFMessage)***__:__<br/>
Transform messages. Used to determine when a transform has been published between the configured coordinate frames, instead of
constantly polling for transforms.

* **doLocOnly** ***(std_msgs/Bool)***__:__<br/>
When received, if set to log errors to a file, it will create a new, separate file for the errors of the Localization-Only phase.

### Published Topics
If set to publish the errors, the following messages will be output.

* **tra_err** ***(std_msgs/Float64)***__:__<br/>
* **rot_err** ***(std_msgs/Float64)***__:__<br/>
* **tot_err** ***(std_msgs/Float64)***__:__<br/>
Translational, rotational and total errors in relation to the ground truth respectively.

### Parameters
* **~lambda** ***(float, default:*** __0.1__***)***__:__<br/>
Weight for the rotational error to compute the total error as:
&#x3B5;_tot = &#x3B5;_tra + &#x3BB; · &#x3B5;_rot

* **~pub_err** ***(bool, default:*** __True__***)***__:__<br/>
Publish the translational, rotational and total errors if set to true.

* **~log_err** ***(bool, default:*** __True__***)***__:__<br/>
Save the translational, rotational and total errors to a file if set to true.

* **~map_frame** ***(str, default:*** __"map"__***)***__:__<br/>
Main reference frame to compare the corrected and ground truth poses from.

* **~odom_frame** ***(str, default:*** __"odom"__***)***__:__<br/>
* **~base_frame** ***(str, default:*** __"base_link"__***)***__:__<br/>
Odometry and Robot base coordinate frames for the corrected poses respectively.

* **~gt_odom_frame** ***(str, default:*** __"GT/odom"__***)***__:__<br/>
* **~gt_base_frame** ***(str, default:*** __"GT/base_link"__***)***__:__<br/>
Odometry and Robot base coordinate frames for the ground truth poses respectively.

* **~log_dir** ***(str, default:*** __"~/Desktop/FMP_logs"__***)***__:__<br/>
Path to the directory in which to store the pose error log files.
* **~err_prefix** ***(str, default:*** __"pose_err"__***)***__:__<br/>
Prefix for automatically constructed filenames. Generated files will be called *\<err_prefix\>\<suffix\>\<ts\>.csv*,
where *\<ts\>* is the execution time in the format *yymmdd_HHMMSS* and *\<suffix\>* is either an empty string for
errors during full-SLAM phase, or *\_loc* during Localization-Only phase.


* **~err_file** ***(str, default:*** __"<\~log_dir\>/\<~err_prefix\>\_\<ts\>.csv"__***)***__:__<br/>
* **~loc_err_file** ***(str, default:*** __"<\~log_dir\>/\<err_prefix\>\_loc\_\<ts\>.csv"__***)***__:__<br/>
Path to the pose error files during Full-SLAM and Localization-Only phases respectively.

* **~newline** ***(str, default:*** __"\n"__***)***__:__
* **~delim** ***(str, default:*** __","__***)***__:__<br/>
Row and column delimiters for the csv file.




# Scripts

## err_curves.py
Script that reads the pose error files stored in a directory and generates curves with error bars.

### Usage:
```commandline
foo@bar:dir$ err_curves.py [-d <dir>] [-x <extension>] [-o <out_dir>] [-m <max_exps>]
```
### Arguments:
* **dir** ***(str, default:*** __"~/Desktop/Experiments/MethodComparison/err"__***)***__:__<br>
Directory where the pose error log files to be plotted are stored.

* **extension** ***(str, default:*** __"csv"__***)***__:__<br>
Extension of the pose error files to be plotted. Used for filtering in case other files are present in the directory.

* **out_dir** ***(str, default:*** __"\<~dir\>/errbar"__***)***__:__<br>
Directory where the error plots are to be saved.

* **max_eps** ***(int, default:*** __0__***)***__:__<br>
Maximum number of experiments to take into consideration for plotting the error curves. Used for limiting the ammount of
data used for cases where the number of experiments varies a lot. If set to 0, then all experiments will be used for plotting.



## err_histograms.py
Script for generating histograms from the pose error files.

### Usage:
```commandline
foo@bar:dir$ err_histograms.py [-d <dir>] [-x <extension>] [-o <out_dir>] [-c <combine_by>]
```

### Arguments:
* **dir** ***(str, default:*** __"~/Desktop/Experiments/MethodComparison/err"__***)***__:__<br>
Directory where the pose error log files to be plotted are stored.

* **extension** ***(str, default:*** __"csv"__***)***__:__<br>
Extension of the pose error files to be plotted. Used for filtering in case other files are present in the directory.

* **out_dir** ***(str, default:*** __"\<~dir\>/hist"__***)***__:__<br>
Directory where the error histograms are to be saved.

* **combine_by** ***(str, default:*** __"map_model"__***)***__:__<br>
Field by which to group the histograms with. Valid values are:
    * ***"n_moves"*** number of measurement steps.
    * ***"m_model"*** map models.
    * ***"p_weight"*** particle weighting methods.
    * ***"imp_pose"*** pose improve methods.
    * ***"err_type"*** error types.
    * ***"test_env"*** test environments.


## method_comparison.py
Script for running experiments multiple times in order to capture statistical information using the simulated 10Loop datasets.
It will run the algorithm for every possible combination of configured steps, map models, particle weighting methods, and pose
improvement methods for a given number of iterations.

### Usage:
```commandline
foo@bar:dir$ method_comparison.py [-i <iterations] [-m <moves>] [-mm <map_models>] [-pw <particle_weights>] [-pi <pose_improve>] [-f <launch file>] [-mp <multi_proc>] [-w <num_workers>] [-p <path>]
```

### Arguments:

* **iterations** ***(int, default:*** __100__***)***__:__<br>
Number of times to run each experiment.

* **moves** ***(str, default:*** __"20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,270,300"__***)***__:__<br>
Comma-separated list of number of moves to run the tests with. Valid ROSBags must exist for each of the selected moves.

* **map_models** ***(str, default:*** __"ref,dec"__***)***__:__<br>
Comma-separated list of map model types. Supported values are:
    * ***ref*** for the Reflection Model.
    * ***dec*** for the Exponential Decay Rate Model.

* **particle_weights** ***(str, default:*** __"cmh,ml"__***)***__:__<br>
Coma-separated list of particle weighting methods. Supported values are:
    * ***cmh*** for Closest Mean Hit Likelihood, the original GMapping method.
    * ***ml*** for the Full Map Posterior Measurement Likelihood method.

* **pose_improve** ***(bool, default:*** __False__***)***__:__<br>
If true, then the pose proposal distribution will be improved using scan matching.

* **launch_file** ***(string, default:*** __"$(find fmp_slam_eval)/launch/experiment.launch"__***)***__:__<br>
Path to the launch file to be executed mutliple times.

* **multi_proc** ***(bool, default:*** __True__***)***__:__<br>
Run multiple processes in parallel if set to true.

* **num_workers** ***(int, default:*** __-1__***)***__:__<br>
Number of workers or processes to execute in parallel if *~multi_proc* is set to True. A value of -1 means to use as many workers as the number of CPU cores available.

* **path** ***(str, default:*** __"~/Desktop/Experiments/MethodComparison"__***)***__:__<br>
Path where the results will be saved.

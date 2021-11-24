# SLAM GMapping

This ROS node is used for building a map of the environment and computing the transform between the */map* and */odom*
frames from laser scan messages (*sensor_msgs/LaserScan*) and the odometry data.

## Topics

### Subscribed Topics

* **tf** ***(tf/tfMessage)***__:__<br/>
Transforms necessary to relate frames for laser, base, and odometry.

* **scan** ***(sensor_msgs/LaserScan)***__:__<br/>
Laser scans to create the map from.

### Published Topics
    
* **map_metadata** ***(nav_msgs/MapMetaData)***__:__<br/>
Get the map data from this topic, which is latched, and updated
periodically.

* **map** ***(nav_msgs/OccupancyGrid)***__:__<br/> 
Get the map data from this topic, which is latched, and updated
periodically

* **~entropy** ***(std_msgs/Float64)***__:__<br/>
Estimate of the entropy of the distribution over the robot's pose (a higher value indicates greater uncertainty).
<span style="color:blue;">New in 1.1.0.</span>

#### Full Map Posterior

---

## Services

* **dynamic_map** ***(nav_msgs/GetMap)***__:__<br/>
Call this service to get the map data

---

## Transforms

### Required Transforms

* ***\<laser_sensor_frame\>*** **⟶ base_link:**<br/>
Usually a fixed value, broadcast periodically by a robot_state_publisher, or a tf static_transform_publisher.

* **base_link ⟶ odom:** <br/>
Usually provided by the odometry system (e.g., the driver for the mobile base)

### Provided tf Transforms
    
* **map → odom:**<br/>
The current estimate of the robot's pose within the map frame

---

## Parameters

### ROS Wrapper
* ~~**~inverted_laser**~~ ~~***(string, default: "false"__***)***__:__~~<br/>
**(REMOVED in 1.1.1; transform data is used instead)** Is the laser right side up (scans are ordered CCW), or upside
down (scans are ordered CW)?

* **~throttle_scans** ***(int, default:*** __1__***)***__:__<br/>
Process 1 out of every this many scans (set it to a higher number to skip more scans)

* **~transform_publish_period** ***(float, default:*** __0.05__***)***__:__<br/>
How long (in seconds) between transform publications. To disable broadcasting transforms, set to 0.

##### Transform Frames
* **~base_frame** ***(string, default:*** __"base_link"__***)***__:__<br/>
The frame attached to the mobile base.

* **~map_frame** ***(string, default:*** __"map"__***)***__:__<br/>
The frame attached to the map.

* **~odom_frame** ***(string, default:*** __"odom"__***)***__:__<br/>
The frame attached to the odometry system.

### Map Size and Resolution
* **~xmin** ***(float, default:*** __-100.0__***)***__:__<br/>
Initial map size (in metres)

* **~ymin** ***(float, default:*** __-100.0__***)***__:__<br/>
Initial map size (in metres)

* **~xmax** ***(float, default:*** __100.0__***)***__:__<br/>
Initial map size (in metres)

* **~ymax** ***(float, default:*** __100.0__***)***__:__<br/>
Initial map size (in metres)

* **~delta** ***(float, default:*** __0.05__***)***__:__<br/>
Resolution of the map (in metres per occupancy grid block)

* **~occ_thresh** ***(float, default:*** __0.25__***)***__:__<br/>
Threshold on gmapping's occupancy values. Cells with greater occupancy are considered occupied (i.e., set to 100 in the
resulting sensor_msgs/LaserScan). New in 1.1.0.

### Scan Matching

* **~kernelSize** ***(int, default:*** __1__***)***__:__<br/>
The kernel in which to look for a correspondence

* **~lstep** ***(float, default:*** __0.05__***)***__:__<br/>
The optimization step in translation

* **~astep** ***(float, default:*** __0.05__***)***__:__<br/>
The optimization step in rotation

* **~iterations** ***(int, default:*** __5__***)***__:__<br/>
The number of iterations of the scanmatcher

* **~sigma** ***(float, default:*** __0.05__***)***__:__<br/>
The sigma used by the greedy endpoint matching

* **~lsigma** ***(float, default:*** __0.075__***)***__:__<br/>
The sigma of a beam used for likelihood computation

* **~lskip** ***(int, default:*** __0__***)***__:__<br/>
Number of beams to skip in each scan. Take only every (n+1)th laser ray for computing a match (0 = take all rays)

##### Scan Matching Sensor Model

* **~maxUrange** ***(float, default:*** __80.0__***)***__:__<br/>
The maximum usable range of the laser. A beam is cropped to this value.

* **~maxRange** ***(float)***__:__<br/>
The maximum range of the sensor. If regions with no obstacles within the range of the sensor should appear as free space
 n the map, set maxUrange \< maximum range of the real sensor ≤ maxRange.
 
##### Scan Matching Sampling Ranges
* **~llsamplerange** ***(float, default:*** __0.01__***)***__:__<br/>
Translational sampling range for the likelihood

* **~llsamplestep** ***(float, default:*** __0.01__***)***__:__<br/>
Translational sampling step for the likelihood

* **~lasamplerange** ***(float, default:*** __0.005__***)***__:__<br/>
Angular sampling range for the likelihood

* **~lasamplestep** ***(float, default:*** __0.005__***)***__:__<br/>
Angular sampling step for the likelihood

### Motion Model
Standard deviations of a Gaussian noise model

* **~srr** ***(float, default:*** __0.1__***)***__:__<br/>
Odometry error in translation as a function of translation (ρ/ρ).<br/>
Linear noise component (*x* and *y*).

* **~stt** ***(float, default:*** __0.2__***)***__:__<br/>
Odometry error in rotation as a function of rotation (θ/θ)<br/>
Angular noise component (θ)

* **~srt** ***(float, default:*** __0.2__***)***__:__<br/>
Odometry error in translation as a function of rotation (ρ/θ)<br/>
Linear ⟶ Angular noise component

* **~str** ***(float, default:*** __0.1__***)***__:__<br/>
Odometry error in rotation as a function of translation (θ/ρ)<br/>
Angular ⟶ Linear noise component

### Update Intervals

* **~map_update_interval** ***(float, default:*** __5.0__***)***__:__<br/>
How long (in seconds) between updates to the map. Lowering this number updates the occupancy grid more often, at the
expense of greater computational load.

* **~linearUpdate** ***(float, default:*** __1.0__***)***__:__<br/>
Process a scan each time the robot translates this far

* **~angularUpdate** ***(float, default:*** __0.5__***)***__:__<br/>
Process a scan each time the robot rotates this far

* **~temporalUpdate** ***(float, default:*** __-1.0__***)***__:__<br/>
Process a scan if the last scan processed is older than the update time in seconds. A value less than zero will turn
time based updates off.

### Particle Filter

* **~particles** ***(int, default:*** __30__***)***__:__<br/>
Number of particles in the filter

* **~resampleThreshold** ***(float, default:*** __0.5__***)***__:__<br/>
The Neff based resampling threshold

* **~minimumScore** ***(float, default:*** __0.0__***)***__:__<br/>
Minimum score for considering the outcome of the scan matching good. Can avoid jumping pose estimates in large open
spaces when using laser scanners with limited range (e.g. 5m). Scores go up to 600+, try 50 for example when
experiencing jumping estimate issues.

* **~ogain** ***(float, default:*** __3.0__***)***__:__<br/>
Gain to be used while evaluating the likelihood, for smoothing the resampling effects

### Full Map Posterior Parameters

* **~en_fmp** ***(bool, default:*** __false__***)***__:__<br/>
Enable Full Map Posterior calculations

* **~decayModel** ***(bool, default:*** __false__***)***__:__<br/>
Use a beam decay map model.

* **~alpha_0** ***(float, default:*** __1.0__***)***__:__<br/>
Value for the alpha parameter of the map's prior probability distribution.
  
  For an uninformed prior distribution, set to 1.0 for either the Reflection or the Decay Map Model.
  
* **~beta_0** ***(float, default:*** __1.0__***)***__:__<br/>
  Value for the beta parameter of the prior map's probability distribution.
  
  For an uninformed prior distribution, set to:
  * 1.0 for the Reflection Map Model
  * 0.0 for the Decay Map Model
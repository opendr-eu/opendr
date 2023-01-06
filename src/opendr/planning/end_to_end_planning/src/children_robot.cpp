// Copyright 2020-2023 OpenDR European Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <signal.h>
#include "ros/ros.h"

#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>

#include <webots_ros/get_float.h>
#include <webots_ros/set_float.h>
#include <webots_ros/set_int.h>

#include <webots_ros/range_finder_get_info.h>
#include <webots_ros/robot_get_device_list.h>
#include <webots_ros/save_image.h>

#include <geometry_msgs/WrenchStamped.h>
#include <webots_ros/BoolStamped.h>
#include <webots_ros/Float64Stamped.h>
#include <webots_ros/StringStamped.h>
#include <webots_ros/get_float_array.h>
#include <webots_ros/get_int.h>
#include <webots_ros/set_string.h>

#define TIME_STEP 32;
using namespace std;

static int controllerCount;
static std::vector<std::string> controllerList;
static std::vector<float> imageRangeFinder;
static double touchSensorValues[3] = {0, 0, 0};
static bool callbackCalled = false;

ros::ServiceClient time_step_client;
webots_ros::set_int time_step_srv;
// catch names of the controllers availables on ROS network
void controllerNameCallback(const std_msgs::String::ConstPtr &name) {
  controllerCount++;
  controllerList.push_back(name->data);
  ROS_INFO("Controller #%d: %s.", controllerCount, controllerList.back().c_str());
}

// get range image from the range-finder
void rangeFinderCallback(const sensor_msgs::Image::ConstPtr &image) {
  int size = image->width * image->height;
  imageRangeFinder.resize(size);

  const float *depth_data = reinterpret_cast<const float *>(&image->data[0]);
  for (int i = 0; i < size; ++i)
    imageRangeFinder[i] = depth_data[i];
}
// touch sensor
void touchSensorCallback(const webots_ros::Float64Stamped::ConstPtr &value) {
  ROS_INFO("Touch sensor sent value %f (time: %d:%d).", value->data, value->header.stamp.sec, value->header.stamp.nsec);
  callbackCalled = true;
}

void touchSensorBumperCallback(const webots_ros::BoolStamped::ConstPtr &value) {
  ROS_INFO("Touch sensor sent value %d (time: %d:%d).", value->data, value->header.stamp.sec, value->header.stamp.nsec);
  callbackCalled = true;
}

void touchSensor3DCallback(const geometry_msgs::WrenchStamped::ConstPtr &values) {
  touchSensorValues[0] = values->wrench.force.x;
  touchSensorValues[1] = values->wrench.force.y;
  touchSensorValues[2] = values->wrench.force.z;

  ROS_INFO("Touch sensor values are x = %f, y = %f and z = %f (time: %d:%d).", touchSensorValues[0], touchSensorValues[1],
           touchSensorValues[2], values->header.stamp.sec, values->header.stamp.nsec);
  callbackCalled = true;
}

void quit(int sig) {
  time_step_srv.request.value = 0;
  time_step_client.call(time_step_srv);
  ROS_INFO("User stopped the 'catch_the_bird' node.");
  ros::shutdown();
  exit(0);
}

int main(int argc, char **argv) {
  std::string controllerName;
  std::vector<std::string> deviceList;
  std::string rangeFinderName;
  std::string touchSensorName;

  int width, height;
  float i, step;

  // create a node named 'range' on ROS network
  ros::init(argc, argv, "range", ros::init_options::AnonymousName);
  ros::NodeHandle n;

  signal(SIGINT, quit);

  // subscribe to the topic model_name to get the list of availables controllers
  ros::Subscriber nameSub = n.subscribe("model_name", 100, controllerNameCallback);
  while (controllerCount == 0 || controllerCount < nameSub.getNumPublishers()) {
    ros::spinOnce();
    ros::spinOnce();
    ros::spinOnce();
  }
  ros::spinOnce();

  // if there is more than one controller available, let the user choose
  if (controllerCount == 1)
    controllerName = controllerList[0];
  else {
    int wantedController = 0;
    std::cout << "Choose the # of the controller you want to use:\n";
    std::cin >> wantedController;
    if (1 <= wantedController && wantedController <= controllerCount)
      controllerName = controllerList[wantedController - 1];
    else {
      ROS_ERROR("Invalid number for  controller choice.");
      return 1;
    }
  }
  // leave topic once it's not necessary anymore
  nameSub.shutdown();
  // call device_list service to get the list of the devices available on the controller and print it the device_list_srv object
  // contains 2 members request and response. Their fields are described in the corresponding .srv file
  ros::ServiceClient deviceListClient =
    n.serviceClient<webots_ros::robot_get_device_list>(controllerName + "/robot/get_device_list");
  webots_ros::robot_get_device_list deviceListSrv;

  if (deviceListClient.call(deviceListSrv))
    deviceList = deviceListSrv.response.list;
  else
    ROS_ERROR("Failed to call service device_list.");
  rangeFinderName = deviceList[1];
  touchSensorName = deviceList[0];
  ros::ServiceClient rangeFinderGetInfoClient =
    n.serviceClient<webots_ros::range_finder_get_info>(controllerName + '/' + rangeFinderName + "/get_info");
  webots_ros::range_finder_get_info rangeFinderGetInfoSrv;
  if (rangeFinderGetInfoClient.call(rangeFinderGetInfoSrv)) {
    width = rangeFinderGetInfoSrv.response.width;
    height = rangeFinderGetInfoSrv.response.height;
    ROS_INFO("Range-finder size is %d x %d.", width, height);
  } else
    ROS_ERROR("Failed to call service range_finder_get_info.");

  // enable the range-finder
  ros::ServiceClient enableRangeFinderClient =
    n.serviceClient<webots_ros::set_int>(controllerName + '/' + rangeFinderName + "/enable");
  webots_ros::set_int enableRangeFinderSrv;
  ros::Subscriber subRangeFinderRangeFinder;

  enableRangeFinderSrv.request.value = 2 * TIME_STEP;
  if (enableRangeFinderClient.call(enableRangeFinderSrv) && enableRangeFinderSrv.response.success) {
    ROS_INFO("Range-finder enabled with sampling period %d.", enableRangeFinderSrv.request.value);
    subRangeFinderRangeFinder = n.subscribe(controllerName + '/' + rangeFinderName + "/range_image", 1, rangeFinderCallback);

    // wait for  the topics to be initialized
    while (subRangeFinderRangeFinder.getNumPublishers() == 0) {
    }
  } else {
    ROS_ERROR("Failed to call service enable for %s.", rangeFinderName.c_str());
  }
  // enable time_step
  time_step_client = n.serviceClient<webots_ros::set_int>(controllerName + "/robot/time_step");
  time_step_srv.request.value = TIME_STEP;

  ///////////////////////////////
  // TOUCH SENSOR //
  ///////////////////////////////

  ros::ServiceClient set_touch_sensor_client;
  webots_ros::set_int touch_sensor_srv;
  ros::Subscriber sub_touch_sensor_32;
  set_touch_sensor_client = n.serviceClient<webots_ros::set_int>(controllerName + "/touch_sensor/enable");

  ros::ServiceClient sampling_period_touch_sensor_client;
  webots_ros::get_int sampling_period_touch_sensor_srv;
  sampling_period_touch_sensor_client =
    n.serviceClient<webots_ros::get_int>(controllerName + "/touch_sensor/get_sampling_period");

  ros::ServiceClient touch_sensor_get_type_client;
  webots_ros::get_int touch_sensor_get_type_srv;
  touch_sensor_get_type_client = n.serviceClient<webots_ros::get_int>(controllerName + "/touch_sensor/get_type");

  touch_sensor_get_type_client.call(touch_sensor_get_type_srv);
  ROS_INFO("Touch_sensor is of type %d.", touch_sensor_get_type_srv.response.value);

  touch_sensor_get_type_client.shutdown();
  time_step_client.call(time_step_srv);

  touch_sensor_srv.request.value = 32;
  if (set_touch_sensor_client.call(touch_sensor_srv) && touch_sensor_srv.response.success) {
    ROS_INFO("Touch_sensor enabled.");
    if (touch_sensor_get_type_srv.response.value == 0)
      sub_touch_sensor_32 = n.subscribe(controllerName + "/touch_sensor/value", 1, touchSensorBumperCallback);
    else if (touch_sensor_get_type_srv.response.value == 1)
      sub_touch_sensor_32 = n.subscribe(controllerName + "/touch_sensor/value", 1, touchSensorCallback);
    else
      sub_touch_sensor_32 = n.subscribe(controllerName + "/touch_sensor/values", 1, touchSensor3DCallback);
    callbackCalled = false;
    while (sub_touch_sensor_32.getNumPublishers() == 0 && !callbackCalled) {
      ros::spinOnce();
      time_step_client.call(time_step_srv);
    }
  } else {
    if (!touch_sensor_srv.response.success)
      ROS_ERROR("Sampling period is not valid.");
    ROS_ERROR("Failed to enable touch_sensor.");
    return 1;
  }

  sub_touch_sensor_32.shutdown();
  time_step_client.call(time_step_srv);

  sampling_period_touch_sensor_client.call(sampling_period_touch_sensor_srv);
  ROS_INFO("Touch_sensor is enabled with a sampling period of %d.", sampling_period_touch_sensor_srv.response.value);

  time_step_client.call(time_step_srv);

  sampling_period_touch_sensor_client.call(sampling_period_touch_sensor_srv);
  sampling_period_touch_sensor_client.shutdown();
  time_step_client.call(time_step_srv);

  // main loop
  while (ros::ok()) {
    if (!time_step_client.call(time_step_srv) || !time_step_srv.response.success) {
      ROS_ERROR("Failed to call next step with time_step service.");
      exit(1);
    }
    ros::spinOnce();
    while (imageRangeFinder.size() < (width * height))
      ros::spinOnce();
  }
  time_step_srv.request.value = 0;
  time_step_client.call(time_step_srv);
  n.shutdown();
}

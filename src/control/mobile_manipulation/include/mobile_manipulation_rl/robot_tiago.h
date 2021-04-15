#pragma once
#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <geometry_msgs/Twist.h>
#include <mobile_manipulation_rl/robot_env.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <pr2_mechanism_msgs/SwitchController.h>
#include <ros/topic.h>
// #include <controller_manager_msgs/SwitchController.h>

typedef actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> TrajClientTiago;
typedef boost::shared_ptr<TrajClientTiago> TiagoClientPtr;

class RobotTiago : public RobotDiffDrive {
private:
  TiagoClientPtr arm_client_;
  TiagoClientPtr torso_client_;
  TiagoClientPtr gripper_client_;
  // ros::ServiceClient switch_controller_client_;
  // moveit::planning_interface::MoveGroupInterface* move_group_arm_torso_;
  control_msgs::FollowJointTrajectoryGoal arm_goal_;
  control_msgs::FollowJointTrajectoryGoal torso_goal_;

  void setup();
  void sendArmCommand(const std::vector<double> &target_joint_values, double exec_duration) override;
  bool getArmSuccess() override;
  // void stop_controllers();
  // void start_controllers();
public:
  RobotTiago(uint32_t seed, std::string strategy, std::string world_type, bool init_controllers, double penalty_scaling,
             double time_step, bool perform_collision_check, std::string node_handle_name, bool verbose,
             std::string robo_conf_path);
  ~RobotTiago() {
    // delete move_group_arm_torso_;
  }

  void openGripper(double position, bool wait_for_result) override;
  void closeGripper(double position, bool wait_for_result) override;
};

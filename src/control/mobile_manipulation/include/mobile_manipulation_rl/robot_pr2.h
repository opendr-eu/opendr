#pragma once
#include <actionlib/client/simple_action_client.h>
#include <geometry_msgs/Twist.h>
#include <mobile_manipulation_rl/robot_env.h>
#include <pr2_controllers_msgs/JointTrajectoryAction.h>
#include <pr2_controllers_msgs/Pr2GripperCommandAction.h>
#include <pr2_mechanism_msgs/SwitchController.h>

// Our Action interface type, provided as a typedef for convenience
typedef actionlib::SimpleActionClient<pr2_controllers_msgs::Pr2GripperCommandAction> GripperClientPR2;
typedef actionlib::SimpleActionClient<pr2_controllers_msgs::JointTrajectoryAction> TrajClientPR2;

class RobotPR2 : public RobotOmniDrive {
private:
  TrajClientPR2 *arm_client_;
  GripperClientPR2 *gripper_client_;
  // ros::ServiceClient switch_controller_client_;
  pr2_controllers_msgs::JointTrajectoryGoal arm_goal_;
  void sendArmCommand(const std::vector<double> &target_joint_values, double exec_duration) override;
  bool getArmSuccess() override;
  // void stop_controllers();
  // void start_controllers();
  void moveGripper(double position, double effort, bool wait_for_result);

public:
  RobotPR2(uint32_t seed, std::string strategy, std::string world_type, bool init_controllers, double penalty_scaling,
           double time_step, bool perform_collision_check, std::string node_handle_name, bool verbose,
           std::string robo_conf_path);

  ~RobotPR2() {
    if (init_controllers_) {
      delete gripper_client_;
      delete arm_client_;
    }
  }

  void openGripper(double position, bool wait_for_result) override;
  void closeGripper(double position, bool wait_for_result) override;
};

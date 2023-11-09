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

#include <gripper_planner/base_gripper_planner.hpp>
#include <gripper_planner/gmm_planner.hpp>
#include <gripper_planner/linear_planner.hpp>
//#include <mobile_manipulation_rl/robot_hsr.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <mobile_manipulation_rl/robot_pr2.hpp>
#include <mobile_manipulation_rl/robot_tiago.hpp>

namespace py = pybind11;

PYBIND11_MODULE(pybindings, m) {
  py::class_<RobotObs>(m, "RobotObs")
    .def(py::init<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>,
                  double, std::vector<double>, bool, double, bool>())
    .def_readonly("base_tf", &RobotObs::base_tf)
    .def_readonly("gripper_tf", &RobotObs::gripper_tf)
    .def_readonly("relative_gripper_tf", &RobotObs::relative_gripper_tf)
    .def_readonly("joint_values", &RobotObs::joint_values)
    .def_readonly("base_velocity", &RobotObs::base_velocity)
    .def_readonly("base_rotation_velocity", &RobotObs::base_rotation_velocity)
    .def_readonly("gripper_velocities", &RobotObs::gripper_velocities)
    .def_readonly("ik_fail", &RobotObs::ik_fail)
    .def_readonly("reward", &RobotObs::reward)
    .def_readonly("done", &RobotObs::done);

  py::class_<EEObs>(m, "EEObs")
    .def(py::init<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, double, bool>())
    .def_readonly("next_gripper_tf", &EEObs::next_gripper_tf)
    .def_readonly("next_base_tf", &EEObs::next_base_tf)
    .def_readonly("ee_velocities_world", &EEObs::ee_velocities_world)
    .def_readonly("ee_velocities_rel", &EEObs::ee_velocities_rel)
    .def_readonly("reward", &EEObs::reward)
    .def_readonly("done", &EEObs::done);

  py::class_<RobotPR2>(m, "RobotPR2")
    .def(py::init<uint32_t, std::string, std::string, bool, double, double, bool, std::string, bool, std::string>())
    .def("step", &RobotPR2::step, "Execute the next time step in environment.")
    .def("get_robot_obs", &RobotPR2::getRobotObs, "getRobotObs.")
    .def("reset", &RobotPR2::reset, "Reset environment.")
    .def("get_obs_dim", &RobotPR2::getObsDim, "Get size of the obs vector.")
    .def("get_init_controllers", &RobotPR2::getInitControllers, "getInitControllers.")
    .def("set_world", &RobotPR2::setWorld, "setWorld.")
    .def("get_world", &RobotPR2::getWorld, "getWorld.")
    .def("is_analytical_world", &RobotPR2::isAnalyticalWorld, "isAnalyticalWorld.")
    .def("open_gripper", &RobotPR2::openGripper, "Open the gripper.")
    .def("close_gripper", &RobotPR2::closeGripper, "Close the gripper.")
    .def("get_robo_config", &RobotPR2::getRoboConfig, "getRoboConfig.")
    .def("tip_to_gripper_tf", &RobotPR2::tipToGripperTf, "tipToGripperTf.")
    .def("gripper_to_tip_tf", &RobotPR2::gripperToTipTf, "gripperToTipTf.")
    .def("world_to_relative_tf", &RobotPR2::worldToRelativeTf, "worldToRelativeTf.")
    .def("publish_marker", &RobotPR2::publishMarker, "publishMarker.");

  py::class_<RobotTiago>(m, "RobotTiago")
    .def(py::init<uint32_t, std::string, std::string, bool, double, double, bool, std::string, bool, std::string>())
    .def("step", &RobotTiago::step, "Execute the next time step in environment.")
    .def("get_robot_obs", &RobotTiago::getRobotObs, "getRobotObs.")
    .def("reset", &RobotTiago::reset, "Reset environment.")
    .def("get_obs_dim", &RobotTiago::getObsDim, "Get size of the obs vector.")
    .def("get_init_controllers", &RobotTiago::getInitControllers, "getInitControllers.")
    .def("set_world", &RobotTiago::setWorld, "setWorld.")
    .def("get_world", &RobotTiago::getWorld, "getWorld.")
    .def("is_analytical_world", &RobotTiago::isAnalyticalWorld, "isAnalyticalWorld.")
    .def("open_gripper", &RobotTiago::openGripper, "Open the gripper.")
    .def("close_gripper", &RobotTiago::closeGripper, "Close the gripper.")
    .def("get_robo_config", &RobotTiago::getRoboConfig, "getRoboConfig.")
    .def("tip_to_gripper_tf", &RobotTiago::tipToGripperTf, "tipToGripperTf.")
    .def("gripper_to_tip_tf", &RobotTiago::gripperToTipTf, "gripperToTipTf.")
    .def("world_to_relative_tf", &RobotTiago::worldToRelativeTf, "worldToRelativeTf.")
    .def("publish_marker", &RobotTiago::publishMarker, "publishMarker.");

  //    py::class_<RobotHSR>(m, "RobotHSR")
  //        .def(py::init<uint32_t,
  //                      std::string,
  //                      std::string,
  //                      bool,
  //                      double,
  //                      double,
  //                      bool,
  //                      std::string,
  //                      bool,
  //                      std::string,
  //                      double,
  //                      double,
  //                      bool>())
  //        .def("step", &RobotHSR::step, "Execute the next time step in environment.")
  //        .def("get_robot_obs", &RobotHSR::getRobotObs, "getRobotObs.")
  //        .def("reset", &RobotHSR::reset, "Reset environment.")
  //        .def("get_obs_dim", &RobotHSR::getObsDim, "Get size of the obs vector.")
  //        .def("get_init_controllers", &RobotHSR::getInitControllers, "getInitControllers.")
  //        .def("set_world", &RobotHSR::setWorld, "setWorld.")
  //        .def("get_world", &RobotHSR::getWorld, "getWorld.")
  //        .def("is_analytical_world", &RobotHSR::isAnalyticalWorld, "isAnalyticalWorld.")
  //        .def("open_gripper", &RobotHSR::openGripper, "Open the gripper.")
  //        .def("close_gripper", &RobotHSR::closeGripper, "Close the gripper.")
  //        .def("tip_to_gripper_tf", &RobotHSR::tipToGripperTf, "tipToGripperTf.")
  //        .def("gripper_to_tip_tf", &RobotHSR::gripperToTipTf, "gripperToTipTf.")
  //        .def("world_to_relative_tf", &RobotHSR::worldToRelativeTf, "worldToRelativeTf.")
  //        .def("publish_marker", &RobotHSR::publishMarker, "publishMarker.")
  //        .def("set_ik_slack", &RobotHSR::setIkSlack, "setIkSlack.")
  //        .def("get_robo_config", &RobotHSR::getRoboConfig, "getRoboConfig.")
  //        .def("set_gripper_goal_wrist", &RobotHSR::setGripperGoalWrist, "setGripperGoalWrist.");

  py::class_<LinearPlanner>(m, "LinearPlanner")
    .def(py::init<const std::vector<double>, const std::vector<double>, const std::vector<double>, const std::vector<double>,
                  double, double, double, double, double, double, double, bool>())
    .def("step", &LinearPlanner::step, "step.")
    .def("generate_obs_step", &LinearPlanner::generateObsStep, "generateObsStep.");

  py::class_<GMMPlanner>(m, "GMMPlanner")
    .def(py::init<const std::vector<double>, const std::vector<double>, const std::vector<double>, const std::vector<double>,
                  double, double, double, double, double, double, double, bool, const std::vector<double>,
                  const std::vector<double>, std::string, double>())
    .def("step", &GMMPlanner::step, "step.")
    .def("generate_obs_step", &GMMPlanner::generateObsStep, "generateObsStep.")
    .def("obj_origin_to_tip", &GMMPlanner::objOriginToTip, "objOriginToTip.");

  m.def("multiply_tfs", &utils::pythonMultiplyTfs, "pythonMultiplyTfs");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}

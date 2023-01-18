from opendr.planning.end_to_end_planning.e2e_planning_learner import EndToEndPlanningRLLearner
import os
if os.environ.get("ROS_DISTRO") == "melodic" or os.environ.get("ROS_DISTRO") == "noetic":
    from opendr.planning.end_to_end_planning.envs.UAV_depth_planning_env import UAVDepthPlanningEnv

__all__ = ['EndToEndPlanningRLLearner', 'UAVDepthPlanningEnv']

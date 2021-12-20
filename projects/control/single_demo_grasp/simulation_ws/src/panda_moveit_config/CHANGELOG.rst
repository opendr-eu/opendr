^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package panda_moveit_config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.7.5 (2021-03-25)
------------------
* Update SRDF for collision model (`#35 <https://github.com/ros-planning/panda_moveit_config/issues/35>`_)
  franka_description got updated with a more coarse collision model [1] matching the internal
  self-collision detection, so the SRDF needs to be adapted.
  Disable collision checking between panda_link6 and panda_link8 which collide in the default pose.
  Tested with example controllers.
  Closes `#18 <https://github.com/ros-planning/panda_moveit_config/issues/18>`_. Resolves `ros-planning/moveit#1210 <https://github.com/ros-planning/moveit/issues/1210>`_, resolves `frankaemika/franka_ros#39 <https://github.com/frankaemika/franka_ros/issues/39>`_.
  [1] https://github.com/frankaemika/franka_ros/commit/e52c03a23aa18c6532e40f9bf4927dedfc0c596a
* Fix ordering of planning adapters (`#69 <https://github.com/ros-planning/panda_moveit_config/issues/69>`_)
  As explained in https://github.com/ros-planning/moveit/pull/2053
  AddTimeParameterization should be at the begining of the list.
* Add tranposrt joint state (`#67 <https://github.com/ros-planning/panda_moveit_config/issues/67>`_)
  Co-authored-by: Libor Wagner <libor.wagner@cvut.cz>
* Bump required cmake version (`#61 <https://github.com/ros-planning/panda_moveit_config/issues/61>`_)
* Contributors: Florian Walch, Libor Wagner, Michael Görner, tnaka

0.7.4 (2020-03-27)
------------------
* [fix] Add time parameterization in stomp_planning_pipeline.launch (`#59 <https://github.com/ros-planning/panda_moveit_config/issues/59>`_)
* [fix] add joint_state_publisher_gui to package.xml (`#54 <https://github.com/ros-planning/panda_moveit_config/issues/54>`_)
* [fix] Add static transform publisher (`#51 <https://github.com/ros-planning/panda_moveit_config/issues/51>`_)
* [maintanence] demo_chomp.launch: reuse demo.launch (`#57 <https://github.com/ros-planning/panda_moveit_config/issues/57>`_)
* [maintanence] Define 'extended' state for 'panda_arm' group (`#47 <https://github.com/ros-planning/panda_moveit_config/issues/47>`_)
* Contributors: Mike Lautman, Robert Haschke, Sebastian Wallkötter, jsbyysheng

0.7.3 (2019-11-21)
------------------
* cleanup warehouse settings (`#43 <https://github.com/ros-planning/panda_moveit_config/issues/43>`_)
* demo_chomp.launch: Rename arg planner to pipeline (`#42 <https://github.com/ros-planning/panda_moveit_config/issues/42>`_)
  Fixup of 0b7ac4e6ea147003fd01d0b51723452ff320a5ea (`#29 <https://github.com/ros-planning/panda_moveit_config/issues/29>`_)
* Remove deprecated inorder tag to fix warning (`#36 <https://github.com/ros-planning/panda_moveit_config/issues/36>`_)
* Necessary files to run LERP (`#40 <https://github.com/ros-planning/panda_moveit_config/issues/40>`_)
  * added new files for emptyplan
  * edited demo.launch to accept planner arg
  * Added emptyplan_planning.yaml so it can be used for new planners. It is an empty file.
  Added rosparam in emptyplan_planning_pipeline.launch.xml to load its yaml file
  * added necessary files for LERP
  * chaged the virtual joint to fixed
  * added yaml file
  * addressed the comments
  * revet back to floating for virtual joint and revert moveit.rviz
* Add fully extended pose: 'extended' (`#39 <https://github.com/ros-planning/panda_moveit_config/issues/39>`_)
* Add TrajOpt launch files to panda_moveit_config (`#29 <https://github.com/ros-planning/panda_moveit_config/issues/29>`_)
  * added new files for emptyplan
  * edited demo.launch to accept planner arg
  * Added emptyplan_planning.yaml so it can be used for new planners. It is an empty file.
  Added rosparam in emptyplan_planning_pipeline.launch.xml to load its yaml file
  * from empty to trajopt
  * modified some comment
  * renamed and edited the context of files from empty to trajopt
  * removed  move_group_trajop.launch, we do not need this file
  * applied Dave's comments
  * restored setup_assistant
  * added trajopt params in yaml file
  removed extra planner arg in demo.launch
  * addressed the comments
  * corrected pipeline in move_group
* Added open and closed poses (`#34 <https://github.com/ros-planning/panda_moveit_config/issues/34>`_)
* Adapt pipelines (`#31 <https://github.com/ros-planning/panda_moveit_config/issues/31>`_)
  * add default_planner_request_adapters/ResolveConstraintFrames
  * chomp pipeline: define default planning adapters
* demo.launch: expose planner arg (`#30 <https://github.com/ros-planning/panda_moveit_config/issues/30>`_)
* Contributors: Dave Coleman, Henning Kayser, Jafar Abdi, Omid Heidari, Robert Haschke, simonGoldstein

0.7.2 (2019-04-22)
------------------
* removing unused attempts param (`#26 <https://github.com/ros-planning/panda_moveit_config/issues/26>`_)
* virtual joint quaternion->rpy
* fixing the virtual joint issue by adding the broadcaster (`#23 <https://github.com/ros-planning/panda_moveit_config/issues/23>`_)
* changing the end effector parent group (`#20 <https://github.com/ros-planning/panda_moveit_config/issues/20>`_)
  * changing the end effector parent group
  * changing virtual joint to floating for use with moveit_visual_tools
* Fix incorrect SRDF path (`#19 <https://github.com/ros-planning/panda_moveit_config/issues/19>`_)
* Contributors: Dave Coleman, Mike Lautman

0.7.1 (2018-12-10)
------------------
* CHOMP: remove "Hybrid" collision detector (`#16 <https://github.com/ros-planning/panda_moveit_config/pull/16>`_)
* Contributors: Robert Haschke

0.7.0 (2018-11-09)
------------------
* Initial release of panda_moveit_config into Melodic, including OMPL, CHOMP and STOMP configs
  We moved/merged this repo from https://github.com/frankaemika/franka_ros.
* Contributors: Dave Coleman, Florian Walch, Mike Lautman, Raghavender Sahdev

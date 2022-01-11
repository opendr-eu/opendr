# OpenDR mobile manipulation demo
<div align="left">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" height="20">
  </a>
</div>

Live demo of mobile manipulation using the [OpenDR toolkit](https://opendr.eu).


## Set-up
Follow the ROS-setup described for the [mobile_manipulation tool](/docs/reference/mobile-manipulation.md). 

## Running the example
Mobile manipulation tasks in the analytical environment can be run as follows:
```bash
python3 mobile_manipulation_demo.py --env pr2
```

Available tasks include `RndStartRndGoals`, `RestrictedWs`, `PickNPlace`, `Door`, `Drawer`. Specific tasks can be specified by adding the `--eval_tasks` flag. By default it will evaluate all tasks sequentially.

The robot can be specified by the `--env` flag to chose between the `PR2` and the `Tiago`. By default this will load a pretrained model.

To run this with controllers in gazebo or the real world, pass in the `--eval_worlds` flag with values `gazebo` or `real_world`. By default it will use the analytical (sim) environment. Note that running on the real robot requires the robot to be set up and to specify goals for the end-effector motions. 

For other options, see the arg flags in `mobile_manipulation_demo.py`.

## Acknowledgement
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871449 (OpenDR). This publication reflects the authors’ views only. The European Commission is not responsible for any use that may be made of the information it contains.

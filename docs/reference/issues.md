# Known Issues

This page includes known issues, compatibility issues as well as possible workarounds.


## Issue: Some ROS nodes have a noticable lag

You should make sure that queue size is set to 1 and the buffer size is large enough to hold the input message.
Even though we have set the appropriate default values for topics in order to avoid this issue, this also depends on your system configuration (e.g., size messages published in input topics).
Be sure to check the discussion and explanatation of this behavior in #275.
Essentially, due to the way ROS handles message a latency of at least 2 frames is expected.


## Issue: Docker image do not fit my embedded device

This can affect several embedded devices, such as NX and TX2, which have limited storage on board.
The easiest solution to this issue is to use external storage (e.g., an sd card or an external SSD).
You can also check the [customization](develop/docs/reference/customize.md) instructions on how you can manually build a docker image that can fit your device.

## Issue: I am trying to install the toolkit on Ubuntu 18.04/20.10/XX.XX, WLS, or any other linux distribution and it doesn't work.

OpenDR toolkit targets native installation on Ubuntu 20.04.
For any other system you are advised to use the docker images that are expected to work on any configuration and operating system.

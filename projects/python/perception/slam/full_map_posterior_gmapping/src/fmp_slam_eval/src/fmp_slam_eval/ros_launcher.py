# Copyright 2020-2023 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import roslaunch
import rosnode

import random
import re

import os
from time import sleep
from fmp_slam_eval.net_utils import next_free_port


class ROSLauncher(object):
    """
    Wrapper class for the ROS Launcher API
    """

    def __init__(self, package, launch_file, wait_for_master=False, log_path=None, monitored_nodes=None,
                 protocol=None, host=None, port=None):
        """
        Constructor for a ROS Launcher object.

        :param package: (str) Name of the package where the .launch file is located.
        :param launch_file: (str) Path to the xml .launch file
        :param wait_for_master: (bool)[Default: False] Waits for a ROS Master to exist if True.
                                      If False, then, if no Master exists, it starts one.
        :param log_path: (str)[Default: None] Path where the log files for the launched nodes will be saved.
                              If None, then they will be saved to the default $ROS_LOG_DIR path
        :param monitored_nodes: (dict)[Default: None] Kill the launch process when the nodes listed here shutdown.
                                      E.g. : {"any": ["kill", "if", "any", "of", "these/nodes", "die"],
                                              "all": ["kill", "if", "all", "of", "these/nodes", "die"]}
        :param protocol: (str)[Default: None] Protocol to start the launcher with. By default it is "http".
        :param host: (str)[Default: None] Host where the launch file will be executed. By default it is "localhost".
        :param port: (int)[Default: None] TCP port where the ROS Master will start. By default 11311.
        """

        self._package = package
        self._launch_file = launch_file

        if log_path:
            os.environ["ROS_LOG_DIR"] = log_path

        self._protocol = "http"
        self._host = "localhost"
        self._port = 11311

        host_changed = False

        if protocol is not None:
            if isinstance(protocol, str):
                self._protocol = protocol
                host_changed = True
            else:
                raise TypeError("Invalid type for a protocol ({}: {}). Only str supported.".format(
                    type(protocol), protocol))

        if host is not None:
            if isinstance(host, str):
                self._host = host
                host_changed = True
            else:
                raise TypeError("Invalid type for a hostname ({}: {}). Only str supported.".format(
                    type(host), host))

        if port is not None:
            if isinstance(port, int):
                if 1024 <= port <= 65535:
                    self._port = port
                    host_changed = True
                else:
                    raise ValueError("Invalid Port number ({}).".format(port))
            elif isinstance(port, str):
                if port.lower() == "auto":
                    random.seed()
                    port = random.randint(1024, 65535)
                    self._port = next_free_port(host=self._host, port=port)
                    host_changed = True
                else:
                    raise ValueError("Invalid option '{}'".format(port))
            else:
                raise TypeError("Invalid type for port ({}: {}). Only int and str supported.".format(type(port), port))

        if host_changed:
            os.environ["ROS_MASTER_URI"] = "{}://{}:{}".format(self._protocol, self._host, self._port)
            os.environ["ROS_HOSTNAME"] = self._host

        self._monitored_nodes = None
        if monitored_nodes is not None:
            if isinstance(monitored_nodes, dict):
                self._monitored_nodes = monitored_nodes
            else:
                raise TypeError("Monitored nodes must be a dictionary of lists.")

        self._uuid = roslaunch.rlutil.get_or_generate_uuid(None, wait_for_master)
        self._launch_obj = None

    def start(self, args_dict=None):
        """
        Start the ROS Launch process and all of the launch file nodes.

        :param args_dict: (dict)[Default: None] Dictionary of arguments for the launch file.

        :return: None
        """

        args_list = []
        if args_dict is not None:
            # launch_args = [self._package, self._launch_file]
            # launch_args = [self._launch_file]
            args_list = ["{}:={}".format(k, v) for k, v in args_dict.items()]
            # launch_args.extend(args_list)

        self._launch_obj = roslaunch.parent.ROSLaunchParent(self._uuid, [(self._launch_file, args_list)])
        self._launch_obj.start()

    def stop(self):
        """
        Shutdown the launch process and all the child nodes and subprocesses with it.

        :return: None
        """

        if self._launch_obj is not None:
            try:
                rosnode.rosnode_cleanup()
            except (rosnode.ROSNodeIOException, rosnode.ROSNodeException) as e:
                print(e)
            self._launch_obj.shutdown()

    def _monitored_nodes_are_dead(self):
        """
        Check if the nodes configured to be monitored are still running.

        :return: (bool) True if the configured nodes are dead. False otherwise.
        """

        if self._monitored_nodes is None:
            return False

        if not self._monitored_nodes:
            return False

        dead_nodes = [re.search("([a-zA-Z_/0-9]*)-[0-9]*", n.name).group(1) for n in self._launch_obj.pm.dead_list]
        nodes_dead = False
        if "any" in self._monitored_nodes:
            nodes_dead = nodes_dead or any((True for n in self._monitored_nodes['any'] if n in dead_nodes))

        if "all" in self._monitored_nodes:
            nodes_dead = nodes_dead or set(self._monitored_nodes['all']).issubset(dead_nodes)

        return nodes_dead

    def is_running(self):
        """
        Return whether the launcher is still running, or if it should/could be shutdown.

        :return: (Bool) False if either the Process Manager is shutdown, the Server is shutdown, or the monitored nodes
                        are dead. True otherwise.
        """

        active_nodes = self._launch_obj.pm.get_active_names()
        pm_is_shutdown = self._launch_obj.pm.is_shutdown
        server_is_shutdown = self._launch_obj.server.is_shutdown
        nodes_dead = self._monitored_nodes_are_dead()
        return not (pm_is_shutdown or server_is_shutdown or len(active_nodes) <= 2 or nodes_dead)

    def spin(self):
        """
        Similar to the ros.spin() method. It just blocks the execution until the is_running method returns false.

        :return: None
        """

        while self.is_running():
            sleep(0.1)

        self.stop()

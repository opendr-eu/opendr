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

import subprocess
import sys
import os
from os import path
import signal
from time import sleep
import psutil

from map_simulator.utils import mkdir_p


class ROSCore(object):
    """
    Wrapper Class for the ROSCore subprocess.
    Used for starting and stopping the ROS master programmatically.
    Useful for automated repeated testing.
    """
    __initialized = False

    def __init__(self, log_dir=""):
        self._proc = None
        self._pid = None
        self._log_dir = log_dir
        self._is_running = False

        if ROSCore.__initialized:
            raise Exception("Only one instance of ROSCore can exist at a time.")
        ROSCore.__initialized = True

    def _wait_for_core(self, sleep_time=0.01, timeout=2):

        elapsed_time = 0

        while elapsed_time <= timeout:
            line = self._proc.stdout.readline()

            if "started core service" in line:
                return

            sleep(sleep_time)
            elapsed_time += sleep_time

        raise OSError("Time-out while waiting for ROSCore to fully start.")

    def start(self):

        env = os.environ.copy()
        if self._log_dir != "":
            if not path.exists(self._log_dir):
                mkdir_p(self._log_dir)
            env['ROS_LOG_DIR'] = self._log_dir

        try:
            self._proc = subprocess.Popen(['roscore'], env=env, stdout=subprocess.PIPE)
            self._pid = self._proc.pid
            self._wait_for_core()
        except OSError as e:
            sys.stderr.write('ROSCore could not be started.')
            raise e

        self._is_running = True

    def _kill_children(self, sig=signal.SIGTERM):
        try:
            parent_process = psutil.Process(self._pid)
        except psutil.NoSuchProcess:
            print("Parent process does not exist.")

        else:
            children_processes = parent_process.children(recursive=True)
            for process in children_processes:
                print("Trying to kill child: " + str(process))
                process.send_signal(sig)

    def stop(self):
        print("Trying to kill child PIDs of ROSCore pid: " + str(self._pid))
        self._kill_children()
        self._proc.terminate()
        self._proc.wait()  # Prevent leaving orphaned zombie processes
        ROSCore.__initialized = False
        self._is_running = False

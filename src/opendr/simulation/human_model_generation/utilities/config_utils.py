# Copyright 2020-2021 OpenDR European Project
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

from opendr.simulation.human_model_generation.utilities.visualizer import Visualizer
from opendr.simulation.human_model_generation.utilities.studio import Studio


def config_visualizer(out_path, mesh, pose=None, plot_kps=False):
    return Visualizer(out_path=out_path, mesh=mesh, pose=pose, plot_kps=plot_kps)


def config_studio():
    return Studio()

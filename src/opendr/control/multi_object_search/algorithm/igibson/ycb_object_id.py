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


import pybullet as p
from igibson.objects.ycb_object import YCBObject


class YCBObject_ID(YCBObject):
    def __init__(self, name, scale=1, **kwargs):
        super(YCBObject_ID, self).__init__(name, scale, **kwargs)

    def _load(self):
        body_id = super(YCBObject_ID, self)._load()
        self.bid = body_id[0]
        return body_id

    def reset(self):
        pass

    def force_wakeup(self):
        """
        Force wakeup sleeping objects
        """
        for joint_id in range(p.getNumJoints(self.get_body_id())):
            p.changeDynamics(self.get_body_id(), joint_id, activationState=p.ACTIVATION_STATE_WAKE_UP)
        p.changeDynamics(self.get_body_id(), -1, activationState=p.ACTIVATION_STATE_WAKE_UP)

# Copyright 2021 - present, OpenDR European Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class World:

  def __init__(self, bullet_client, gravity, world_file, timestep, frame_skip, num_solver_iterations, contact_erp):
    self._p = bullet_client
    self.gravity = gravity
    self.world_file = world_file
    self.timestep = timestep
    self.frame_skip = frame_skip
    self.numSolverIterations = num_solver_iterations
    self.ContactERP = contact_erp
    self.clean_everything()
    flags = 0
    self._p.loadURDF(world_file,
                     basePosition=[0, 0, 0],
                     baseOrientation=[0, 0, 0, 1],
                     useFixedBase=True,
                     flags=flags)

  def clean_everything(self):
    self._p.setGravity(0, 0, -self.gravity)
    self._p.setDefaultContactERP(self.ContactERP)
    self._p.setPhysicsEngineParameter(fixedTimeStep=self.timestep * self.frame_skip,
                                      numSolverIterations=self.numSolverIterations,
                                      numSubSteps=self.frame_skip)

  def step(self):
    self._p.stepSimulation()

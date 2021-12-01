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

import ctypes
import math
import pyglet
import numpy as np
from PIL import Image
from opendr.engine.target import Pose
from opendr.perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from opendr.simulation.human_model_generation.utilities.joint_extractor import Joint_extractor


class Studio(pyglet.window.Window):
    def __init__(self):
        self.image_width = 360
        self.image_height = 720
        super(Studio, self).__init__(width=self.image_width, height=self.image_height, resizable=True)
        self.human_pitch_thres = [0, 15, 5]
        self.human_yaw_thres = [0, 360, 45]
        self.cam_dist_thres = [8.0, 9.0, 1.0]

        self.human_rot = np.asarray([0, 0, 0]).astype(np.float)
        self.human_tr = np.asarray([0, 0, 0]).astype(np.float)
        self.cam_rot = np.asarray([0, 0, 0]).astype(np.float)
        self.cam_tr = np.asarray([0, 0, 0]).astype(np.float)
        self.lightfv = ctypes.c_float * 4
        self.y_tr_offset = 1.0
        self.kps3D_pos = []
        self.box3D_pos = []
        self.joints_computed = False
        self.pose_estimator = LightweightOpenPoseLearner(device="cuda", num_refinement_stages=2,
                                                         mobilenet_use_stride=False, half_precision=False)
        self.pose_estimator.download(path=".", verbose=True)
        self.pose_estimator.load("openpose_default")
        self.run_update = False

    def infer(self, model_3D, mode='normal'):
        self.run_update = True
        self.joints_computed = False
        self.model_3D = model_3D
        flat_verts = np.concatenate(self.model_3D.verts[self.model_3D.faces]).ravel().tolist()
        flat_colors = np.concatenate(self.model_3D.vert_colors[self.model_3D.faces]).ravel().tolist()
        self.verts = pyglet.graphics.vertex_list(3 * len(self.model_3D.verts[model_3D.faces]), ('v3f', flat_verts),
                                                 ('c3f', flat_colors))
        self.filename = 'aaa'
        self.make_plan(mode)
        self.joints_computed = False
        self.kps3D_pos = []
        self.box3D_pos = []
        self.joints_extractor = Joint_extractor()
        pyglet.clock.schedule(self.update)
        pyglet.app.run()
        pyglet.clock.unschedule(self.update)
        self.run_update = False

    def on_resize(self, width, height):
        viewport_width, viewport_height = self.get_framebuffer_size()
        pyglet.gl.glViewport(0, 0, viewport_width, viewport_height)
        pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
        pyglet.gl.glLoadIdentity()
        angleOfView = 45.0
        imageAspectRatio = float(width) / height
        near = 1.
        far = 20.
        pyglet.gl.gluPerspective(angleOfView, imageAspectRatio, near, far)
        pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)
        return True

    def move_camera(self):
        self.cam_tr[0] = 0
        self.cam_tr[1] = -math.sin(math.pi * self.cam_rot[0] / 180.0) * self.cam_dist + 1
        self.cam_tr[2] = -math.cos(math.pi * self.cam_rot[0] / 180.0) * self.cam_dist
        pyglet.gl.glLoadIdentity()
        pyglet.gl.glRotatef(self.cam_rot[0], 1.0, 0.0, 0.0)
        pyglet.gl.glRotatef(self.cam_rot[2], 0.0, 0.0, 1.0)
        pyglet.gl.glTranslatef(self.cam_tr[0], self.cam_tr[1] - self.y_tr_offset, self.cam_tr[2])
        pyglet.gl.glRotatef(self.cam_rot[1], 0.0, 1.0, 0.0)

    def on_draw(self):
        self.clear()
        pyglet.gl.glLoadIdentity()
        pyglet.gl.glLightfv(pyglet.gl.GL_LIGHT0, pyglet.gl.GL_AMBIENT, self.lightfv(0.2, 0.2, 0.2, 1.0))
        pyglet.gl.glEnable(pyglet.gl.GL_COLOR_MATERIAL)
        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)
        pyglet.gl.glShadeModel(pyglet.gl.GL_SMOOTH)
        pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)
        self.move_camera()
        self.draw_model()

        img = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        img = Image.frombytes('RGB', (360, 720), img.get_data('RGB', img.width * len('RGB')), 'raw')
        img = np.array(img)[::-1, :, ::-1]

        poses = self.pose_estimator.infer(img, track=False, smooth=False)
        if len(poses) > 0:
            self.kpt_names = poses[0].kpt_names
            joints_2D = poses[0].data
            self.joints_extractor.compute_rays(joints_2D, self.image_width, self.image_height)

    def draw_model(self):
        pyglet.gl.glPushMatrix()
        pyglet.gl.glRotatef(self.human_rot[0], 1.0, 0.0, 0.0)
        pyglet.gl.glRotatef(self.human_rot[1], 0.0, 1.0, 0.0)
        self.verts.draw(pyglet.gl.GL_TRIANGLES)
        pyglet.gl.glPopMatrix()

    def make_plan(self, mode):

        if mode == 'fast':
            self.current_cam_rot_thres = [[10, 10, 1], [0, 361, 45], [0, 0, 1]]
            self.current_cam_dist_thres = [3.0, 3.0, 1.0]
        else:
            self.current_cam_rot_thres = [[-25, 25, 25], [0, 360, 45], [0, 0, 1]]
            self.current_cam_dist_thres = [2.5, 3.0, 0.5]

        self.current_human_rot_thres = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

        self.current_cam_dist = []
        self.current_cam_rot = []
        self.current_human_rot = []
        for dist_cam in np.arange(self.current_cam_dist_thres[0], self.current_cam_dist_thres[1] + 0.01,
                                  self.current_cam_dist_thres[2]):
            for pitch_cam in np.arange(self.current_cam_rot_thres[0][0], self.current_cam_rot_thres[0][1] + 0.01,
                                       self.current_cam_rot_thres[0][2]):
                for yaw_cam in np.arange(self.current_cam_rot_thres[1][0], self.current_cam_rot_thres[1][1] + 0.01,
                                         self.current_cam_rot_thres[1][2]):
                    for roll_cam in np.arange(self.current_cam_rot_thres[2][0], self.current_cam_rot_thres[2][1] + 0.01,
                                              self.current_cam_rot_thres[2][2]):
                        for pitch_human in np.arange(self.current_human_rot_thres[0][0],
                                                     self.current_human_rot_thres[0][1] + 0.01,
                                                     self.current_human_rot_thres[0][2]):
                            for yaw_human in np.arange(self.current_human_rot_thres[1][0],
                                                       self.current_human_rot_thres[1][1] + 0.01,
                                                       self.current_human_rot_thres[1][2]):
                                for roll_human in np.arange(self.current_human_rot_thres[2][0],
                                                            self.current_human_rot_thres[2][1] + 0.01,
                                                            self.current_human_rot_thres[2][2]):
                                    self.current_cam_dist.append(dist_cam)
                                    self.current_cam_rot.append([pitch_cam, yaw_cam, roll_cam])
                                    self.current_human_rot.append([pitch_human, yaw_human, roll_human])
        self.current_num_shots = len(self.current_cam_dist)

    def compute_3D_box(self, kps):
        if len(kps) == 0:
            return []
        kps_np = np.asarray(kps)
        extra_x = [0.06, 0.06]
        extra_y = [0.1, 0.175]
        extra_z = [0.05, 0.05]

        x_max = np.max(kps_np[:, 0]) + extra_x[0]
        x_min = np.min(kps_np[:, 0]) - extra_x[1]
        y_max = np.max(kps_np[:, 1]) + extra_y[0]
        y_min = np.min(kps_np[:, 1]) - extra_y[1]
        z_max = np.max(kps_np[:, 2]) + extra_z[0]
        z_min = np.min(kps_np[:, 2]) - extra_z[1]
        bbox3D = []
        bbox3D.append(np.array([x_min, y_min, z_min]))
        bbox3D.append(np.array([x_min, y_max, z_min]))
        bbox3D.append(np.array([x_max, y_max, z_min]))
        bbox3D.append(np.array([x_max, y_min, z_min]))
        bbox3D.append(np.array([x_min, y_min, z_max]))
        bbox3D.append(np.array([x_min, y_max, z_max]))
        bbox3D.append(np.array([x_max, y_max, z_max]))
        bbox3D.append(np.array([x_max, y_min, z_max]))
        return bbox3D

    def update(self, dt):
        if self.run_update:
            if len(self.current_cam_dist) > 0:
                self.human_rot = self.current_human_rot.pop(0)
                self.cam_rot = self.current_cam_rot.pop(0)
                self.cam_dist = self.current_cam_dist.pop(0)
            else:
                self.kps3D_pos, self.kps3D_dists = self.joints_extractor.compute_3D_positions
                self.pose_3D = Pose(self.kps3D_pos, np.array([1] * 18))
                self.pose_3D.id = 0
                self.box3D_pos = self.compute_3D_box(self.kps3D_pos)
                self.dict_save_bbox3D = dict([(i, self.box3D_pos[i]) for i in range(len(self.box3D_pos))])
                self.joints_computed = True
                self.joints_extractor = None
                self.close()

    def get_bbox(self):
        if self.joints_computed:
            return self.dict_save_bbox3D

    def get_poses(self):
        if self.joints_computed:
            return self.pose_3D

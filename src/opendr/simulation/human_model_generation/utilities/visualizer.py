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
import os
import argparse
import pyglet
import numpy as np
from PIL import Image as PIL_image
from opendr.engine.data import Image
import cv2


class Visualizer(pyglet.window.Window):

    def __init__(self, out_path, mesh, pose=None, plot_kps=False):
        super().__init__(width=360, height=640, visible=True, resizable=True)
        self.out_path = out_path
        self.mesh = mesh
        flat_verts = np.concatenate(self.mesh.verts[self.mesh.faces]).ravel().tolist()
        flat_colors = np.concatenate(self.mesh.vert_colors[self.mesh.faces]).ravel().tolist()
        self.verts = pyglet.graphics.vertex_list(3 * len(self.mesh.verts[mesh.faces]), ('v3f', flat_verts),
                                                 ('c3f', flat_colors))
        self.lightfv = ctypes.c_float * 4
        self.exit = 0
        self.rotation_id = 0
        self.rotations = [0]
        self.imgs = []
        self.width = 360
        self.height = 640
        self.pose = None
        self.project_pose = False
        if pose is not None:
            self.project_pose = True
            self.pose = pose
        self.imgs = []
        self.kps2D = []
        self.plot_kps = plot_kps

    def infer(self, rotations):
        self.imgs = []
        self.kps2D = []
        self.rotation_id = 0
        self.exit = 0
        self.rotations = rotations
        pyglet.clock.schedule(self.exit_callback)
        pyglet.app.run()
        pyglet.clock.unschedule(self.exit_callback)
        if self.project_pose:
            return [self.imgs, self.kps2D]
        else:
            return self.imgs

    def on_resize(self, width, height):
        viewport_width, viewport_height = self.get_framebuffer_size()
        pyglet.gl.glViewport(0, 0, viewport_width, viewport_height)
        pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
        pyglet.gl.glLoadIdentity()
        pyglet.gl.gluPerspective(45., float(width) / height, 1., 600.)
        pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)
        return True

    def draw_human_model(self):
        pyglet.gl.glPushMatrix()
        pyglet.gl.glTranslated(0, 0, -3.5)
        pyglet.gl.glRotatef(self.rotations[self.rotation_id], 0.0, 1.0, 0.0)
        self.verts.draw(pyglet.gl.GL_TRIANGLES)
        if self.pose is not None:
            kps2D = {}
            pmat = (pyglet.gl.GLdouble * 16)()
            mvmat = (pyglet.gl.GLdouble * 16)()
            view = (pyglet.gl.GLint * 4)()
            pyglet.gl.glGetDoublev(pyglet.gl.GL_MODELVIEW_MATRIX, mvmat)
            pyglet.gl.glGetDoublev(pyglet.gl.GL_PROJECTION_MATRIX, pmat)
            pyglet.gl.glGetIntegerv(pyglet.gl.GL_VIEWPORT, view)
            for i in range(len(self.pose.data)):
                win_x = pyglet.gl.GLdouble()
                win_y = pyglet.gl.GLdouble()
                win_z = pyglet.gl.GLdouble()
                pyglet.gl.gluProject(self.pose.data[i][0], self.pose.data[i][1], self.pose.data[i][2], mvmat, pmat, view,
                                     win_x, win_y, win_z)
                kps2D[i] = np.asarray([win_x.value, self.height - win_y.value, win_z.value]).astype(np.int)
            self.kps2D.append(kps2D)
        pyglet.gl.glPopMatrix()

    def on_draw(self):
        self.clear()
        pyglet.gl.glLoadIdentity()
        pyglet.gl.glLightfv(pyglet.gl.GL_LIGHT0, pyglet.gl.GL_AMBIENT, self.lightfv(0.2, 0.2, 0.2, 1.0))
        pyglet.gl.glEnable(pyglet.gl.GL_COLOR_MATERIAL)
        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)
        pyglet.gl.glShadeModel(pyglet.gl.GL_SMOOTH)
        pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)
        self.draw_human_model()
        self.exit = self.exit + 1

    def exit_callback(self, dt):
        if self.exit > 0:
            img = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            img = PIL_image.frombytes('RGB', (self.width, self.height), img.get_data('RGB', img.width * len('RGB')), 'raw')
            img = np.array(img)[::-1, :, ::-1].astype(np.float32)
            if self.plot_kps:
                for kp in self.kps2D[self.rotation_id]:
                    img = cv2.circle(img, (self.kps2D[self.rotation_id][kp][0], self.kps2D[self.rotation_id][kp][1]), 1,
                                     (0, 1.0, 0), 2)
            self.imgs.append(Image(img))
            self.rotation_id = self.rotation_id + 1
            if self.rotation_id >= len(self.rotations):
                self.close()
            else:
                self.exit = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, default='./')
    parser.add_argument('-o', '--out', type=str, default='./')
    parser.add_argument('-r', '--rotation', type=float, default=0)
    parser.add_argument('-f', '--filename', type=str, default='aa.obj')
    args = parser.parse_args()
    full_path_input = os.path.join(args.input_folder, args.filename + ".obj")
    vis = Visualizer(full_path_input, args.out, args.rotation)
    pyglet.clock.schedule(vis.exit_callback)
    pyglet.app.run()

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


import numpy as np
from pyglet.window import pyglet
import math
import pyglet.gl
import os
from PIL import Image
import cv2


class Background:
    def __init__(self, img_dir, img_fl, width, height, xpos, ypos, zpos, placement_colors):
        self.xpos = xpos
        self.ypos = ypos
        self.zpos = zpos
        self.labels_cls = placement_colors
        self.angle = 0
        self.size = 1
        self.width = width
        self.height = height
        x = width / 2.0
        y = height / 2.0
        self.vlist = pyglet.graphics.vertex_list(4, ('v2f', [-x, -y, x, -y, -x, y, x, y]),
                                                 ('t2f', [0, 0, 1, 0, 0, 1, 1, 1]))
        self.img_dir = img_dir
        self.img_fl = img_fl
        self.select_texture()

    def select_human_pos(self, models_data):
        humans_pos2D = []
        msks = []
        for i in range(len(self.labels_cls)):
            msk = cv2.inRange(self.labels, self.labels_cls[i] - 1, self.labels_cls[i] + 1)
            msks.append(msk)
        msk = msks[0]
        for i in range(len(self.labels_cls)):
            msk = cv2.bitwise_or(msk, msks[i])
        msk_new = np.zeros((msk.shape))
        for i1 in range(msk.shape[0]):
            for i2 in range(msk.shape[1]):
                ok_pix = True
                for j1 in range(-6, 7, 4):
                    for j2 in range(-6, 7, 4):
                        if ((i1 + j1) > 0) & ((i2 + j2) > 0) & ((i1 + j1) < msk.shape[0]) & ((i2 + j2) < msk.shape[1]):
                            if msk[i1 + j1][i2 + j2] == 0:
                                ok_pix = False
                if ok_pix:
                    msk_new[i1][i2] = 255
        ids = np.where(msk_new == 255)
        pixs = list(zip(ids[0], ids[1]))
        pixs = np.array(pixs)
        pixs = pixs[(pixs[:, 0] > 50) & (pixs[:, 0] < (self.labels.shape[0] - 50)) & (pixs[:, 1] > 80) & (
                pixs[:, 1] < (self.labels.shape[1] - 80))]

        for j in range(len(models_data)):
            id_l = 0
            min_dist = 10000000000
            pref_coords = [models_data[j]['img_pos_pref'][0] * self.img.size[1],
                           models_data[j]['img_pos_pref'][1] * self.img.size[0]]
            for k in range(pixs.shape[0]):
                dist = math.sqrt(
                    (pixs[k][0] - pref_coords[0]) * (pixs[k][0] - pref_coords[0]) + (pixs[k][1] - pref_coords[1]) * (
                            pixs[k][1] - pref_coords[1]))
                if dist < min_dist:
                    id_l = k
                    min_dist = dist
            pos2D = pixs[id_l]
            humans_pos2D.append(pos2D)
            pixs_new = []
            for k in range(pixs.shape[0]):
                if not ((pixs[k][0] > (pos2D[0] - 125)) & (pixs[k][1] > (pos2D[1] - 70)) & (
                        pixs[k][0] < (pos2D[0] + 125)) & (pixs[k][1] < (pos2D[1] + 70))):
                    pixs_new.append(pixs[k])
            pixs = np.array(pixs_new)
        for j in range(len(models_data)):
            humans_pos2D[j] = np.array([humans_pos2D[j][1] / self.labels.shape[1],
                                        (self.labels.shape[0] - humans_pos2D[j][0]) / self.labels.shape[0]])
            models_data[j]['img_coords'] = humans_pos2D[j]
            dst = abs(humans_pos2D[j][0] - 0.5) * 0.05 + humans_pos2D[j][1] * 0.975
            if dst < 0.1:
                d = self.zpos + 27.0
            elif dst < 0.15:
                d = self.zpos + 26.0
            elif dst < 0.2:
                d = self.zpos + 25.0
            elif dst < 0.25:
                d = self.zpos + 24.0
            elif dst < 0.3:
                d = self.zpos + 23.5
            elif dst < 0.35:
                d = self.zpos + 23.0
            elif dst < 0.4:
                d = self.zpos + 22.0
            elif dst < 0.45:
                d = self.zpos + 21.0
            else:
                d = self.zpos + 18.0
            humans_pos2D[j][0] = humans_pos2D[j][0] * self.width - self.width / 2.0
            humans_pos2D[j][1] = humans_pos2D[j][1] * self.height - self.height / 2.0
            models_data[j]['pos_2D'] = np.array([humans_pos2D[j][0], humans_pos2D[j][1]])
            models_data[j]['pos_3D'] = np.zeros(3)
            models_data[j]['pos_3D'][2] = d

    def set_dist(self, z_pos):
        self.zpos = z_pos

    def draw(self):
        pyglet.gl.glPushMatrix()
        pyglet.gl.glTranslatef(self.xpos, self.ypos, self.zpos, 0)
        pyglet.gl.glScalef(self.size, self.size, self.size)
        pyglet.gl.glColor3f(1, 1, 1)
        pyglet.gl.glEnable(pyglet.gl.GL_TEXTURE_2D)
        pyglet.gl.glBindTexture(pyglet.gl.GL_TEXTURE_2D, self.texture)
        self.vlist.draw(pyglet.gl.GL_TRIANGLE_STRIP)
        pyglet.gl.glDisable(pyglet.gl.GL_TEXTURE_2D)
        pyglet.gl.glPopMatrix()

    def get_pos(self):
        return self.xpos, self.ypos, self.z_pos

    def select_texture(self):
        texture_file = os.path.join(self.img_dir, 'rgb', self.img_fl)
        labels_file = os.path.join(self.img_dir, 'segm', self.img_fl)
        print(self.img_fl)
        self.texture = self.loadTexture(texture_file)
        self.labels = cv2.imread(labels_file)
        new_size = (1020, 340)
        self.labels = cv2.resize(self.labels, new_size)
        # self.labels_cls = [np.array([128, 64, 128]), np.array([192, 0, 0])] #kitti
        # self.labels_cls = [np.array([128, 64, 128]), np.array([244, 35, 232]), np.array([152, 251, 152])] #kitti

    def loadTexture(self, filename):
        self.img = Image.open(filename).transpose(Image.FLIP_TOP_BOTTOM)
        new_size = (1020, 340)
        self.img = self.img.resize(new_size)
        textureIDs = (pyglet.gl.GLuint * 1)()
        pyglet.gl.glGenTextures(1, textureIDs)
        textureID = textureIDs[0]
        # print('generating texture', textureID, 'from', filename)
        pyglet.gl.glBindTexture(pyglet.gl.GL_TEXTURE_2D, textureID)
        pyglet.gl.glTexParameterf(pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_WRAP_S, pyglet.gl.GL_REPEAT)
        pyglet.gl.glTexParameterf(pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_WRAP_T, pyglet.gl.GL_REPEAT)
        pyglet.gl.glTexParameterf(pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_MAG_FILTER, pyglet.gl.GL_NEAREST)
        pyglet.gl.glTexParameterf(pyglet.gl.GL_TEXTURE_2D, pyglet.gl.GL_TEXTURE_MIN_FILTER, pyglet.gl.GL_NEAREST)
        pyglet.gl.glTexImage2D(pyglet.gl.GL_TEXTURE_2D, 0, pyglet.gl.GL_RGB, self.img.size[0], self.img.size[1],
                               0, pyglet.gl.GL_RGB, pyglet.gl.GL_UNSIGNED_BYTE, self.img.tobytes())
        pyglet.gl.glBindTexture(pyglet.gl.GL_TEXTURE_2D, 0)
        return textureID

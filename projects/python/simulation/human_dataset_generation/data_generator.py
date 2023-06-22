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

import ctypes
import os
import math
import pyglet.gl
from pywavefront import visualization, Wavefront
import numpy as np
from pyglet.window import pyglet
import pickle
import cv2
import csv
from background import Background


class DataGenerator(pyglet.window.Window):
    def __init__(self, models_dir, background_dir, csv_dt_path='', model_dict_path='', back_imgs_dict_path='',
                 data_out_dir=None, csv_tr_path=None, placement_colors=[]):
        super(DataGenerator, self).__init__(resizable=True)
        self.camera_size = (1920, 640)
        self.set_size(self.camera_size[0], self.camera_size[1])
        self.background_dir = background_dir
        self.csv_name = os.path.splitext(os.path.basename(csv_dt_path))[0]
        self.apply_transf = False
        [self.background_img_fl, self.models_data, self.split] = self.csv_dt_parser(csv_dt_path)
        if not (csv_tr_path is None):
            self.apply_transf = True
            self.csv_tr_parser(csv_tr_path)
        with open(model_dict_path, 'rb') as pkl_file:
            model_ids = pickle.load(pkl_file)
        with open(back_imgs_dict_path, 'rb') as pkl_file:
            back_img_ids = pickle.load(pkl_file)
        self.assign_ids(model_ids, back_img_ids)
        # self.background_dir = os.path.join(self.background_dir,self.split)
        self.models_dir = models_dir
        self.load_meshes_and_data()
        # models_data[x] = {'filename', 'pitch', 'yaw', 'img_pos}
        self.lightfv = ctypes.c_float * 4
        self.y_tr_offset = 1.0
        self.background = Background(self.background_dir, self.background_img_fl, 81, 25, 0, 0, -8, placement_colors)
        print("Proccessing: " + self.csv_name + "...")
        self.cam_dist = 3.5
        self.background.set_dist(self.cam_dist - 33.5)
        self.background.select_human_pos(self.models_data)
        self.first = True
        self.shot_ok = 0
        self.export_data = False
        if not (data_out_dir is None):
            self.export_data = True
            self.data_out_dir = data_out_dir
            if not os.path.exists(self.data_out_dir):
                os.mkdir(self.data_out_dir)
            self.data_out_dir = os.path.join(data_out_dir, self.split)
            if not os.path.exists(self.data_out_dir):
                os.mkdir(self.data_out_dir)

    def assign_ids(self, model_ids, back_img_ids):
        for i in range(len(self.models_data)):
            for j in range(len(model_ids)):
                if model_ids[j]['filename'] == self.models_data[i]['filename']:
                    self.models_data[i]['id'] = model_ids[j]['id']
        for j in range(len(back_img_ids)):
            if self.background_img_fl == back_img_ids[j]['filename']:
                self.background_img_id = back_img_ids[j]['id']
                print(self.background_img_id)

    def csv_dt_parser(self, csv_path):
        data = []
        models_data = []
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0] == 'background':
                    background_img_fl = row[1]
                elif row[0] == 'model':
                    img_pref_pos = np.array([float(row[3]), float(row[2])])
                    pitch = int(row[4])
                    yaw = int(row[5])
                    rot = {
                        'roll': 0,
                        'pitch': pitch,
                        'yaw': yaw
                    }
                    extra_transl = np.array([0, 0, 0])
                    extra_rot = {
                        'roll': 0,
                        'pitch': pitch,
                        'yaw': yaw
                    }
                    transf = {
                        "translation": extra_transl,
                        "rotation": extra_rot
                    }
                    model_data = {
                        'filename': row[1],
                        'rotation': rot,
                        'transformation': transf,
                        'img_pos_pref': img_pref_pos
                    }
                    models_data.append(model_data)
                elif row[0] == 'split':
                    split = row[1]
                else:
                    print("Wrong data format...")
                    exit()
        data.append(background_img_fl)
        data.append(models_data)
        data.append(split)
        return data

    def csv_tr_parser(self, csv_path):
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                for i in range(len(self.models_data)):
                    if row[0] == self.models_data[i]['filename']:
                        self.models_data[i]['transformation']['translation'] = np.array(
                            [float(row[1]), - float(row[2]), -float(row[3])])
                        # self.models_data[i]['rotation']['roll'] = int(row[4])
                        self.models_data[i]['transformation']['rotation']['pitch'] = int(row[5])
                        self.models_data[i]['transformation']['rotation']['yaw'] = int(row[6])

    def on_resize(self, width, height):
        viewport_width, viewport_height = self.get_framebuffer_size()
        pyglet.gl.glViewport(0, 0, viewport_width, viewport_height)
        pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
        pyglet.gl.glLoadIdentity()
        angleOfView = 45.0
        imageAspectRatio = float(width) / height
        near = 1.
        far = 40.
        pyglet.gl.gluPerspective(angleOfView, imageAspectRatio, near, far)
        pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)
        return True

    def move_camera(self):
        pyglet.gl.glPushMatrix()
        cam_tr = np.zeros(3)
        cam_rot = np.zeros(3)
        cam_tr[0] = 0
        cam_tr[1] = -math.sin(0) * self.cam_dist + 1
        cam_tr[2] = -math.cos(0) * self.cam_dist
        pyglet.gl.glLoadIdentity()
        pyglet.gl.glTranslatef(cam_tr[0], cam_tr[1] - self.y_tr_offset, cam_tr[2])
        pyglet.gl.glRotatef(cam_rot[2], 0.0, 0.0, 1.0)
        pyglet.gl.glRotatef(cam_rot[0], 1.0, 0.0, 0.0)
        pyglet.gl.glRotatef(cam_rot[1], 0.0, 1.0, 0.0)
        pyglet.gl.glPopMatrix()

    def load_meshes_and_data(self):
        for i in range(len(self.models_data)):
            obj_path = os.path.join(self.models_dir, self.models_data[i]['filename'], 'data',
                                    self.models_data[i]['filename'] + '.obj')
            j3D_path = os.path.join(self.models_dir, self.models_data[i]['filename'], 'data',
                                    self.models_data[i]['filename'] + '_j3D.pkl')
            bbox3D_path = os.path.join(self.models_dir, self.models_data[i]['filename'], 'data',
                                       self.models_data[i]['filename'] + '_bbox3D.pkl')
            mesh = Wavefront(obj_path)
            with open(j3D_path,
                      'rb') as pkl_file:
                j3D = pickle.load(pkl_file)
            with open(bbox3D_path,
                      'rb') as pkl_file:
                box3D = pickle.load(pkl_file)
            self.models_data[i]['mesh'] = mesh
            self.models_data[i]['joints_3D'] = j3D
            self.models_data[i]['box_3D'] = box3D

    def on_draw(self):
        self.clear()
        pyglet.gl.glLoadIdentity()
        pyglet.gl.glLightfv(pyglet.gl.GL_LIGHT0, pyglet.gl.GL_POSITION, self.lightfv(-1.0, 1.0, 1.0, 0.0))
        pyglet.gl.glLightfv(pyglet.gl.GL_LIGHT0, pyglet.gl.GL_AMBIENT, self.lightfv(0.2, 0.2, 0.2, 1.0))
        pyglet.gl.glLightfv(pyglet.gl.GL_LIGHT0, pyglet.gl.GL_DIFFUSE, self.lightfv(0.5, 0.5, 0.5, 1.0))
        pyglet.gl.glEnable(pyglet.gl.GL_LIGHT0)
        pyglet.gl.glEnable(pyglet.gl.GL_LIGHTING)
        pyglet.gl.glEnable(pyglet.gl.GL_COLOR_MATERIAL)
        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)
        pyglet.gl.glShadeModel(pyglet.gl.GL_SMOOTH)
        pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)
        self.move_camera()
        self.background.draw()
        for i in range(len(self.models_data)):
            self.draw_model()
        self.shot_ok = self.shot_ok + 1

    def draw_model(self):

        if self.first:
            for i in range(len(self.models_data)):
                near_x = pyglet.gl.GLdouble()
                near_y = pyglet.gl.GLdouble()
                near_z = pyglet.gl.GLdouble()
                start_x = pyglet.gl.GLdouble()
                start_y = pyglet.gl.GLdouble()
                start_z = pyglet.gl.GLdouble()
                end_x = pyglet.gl.GLdouble()
                end_y = pyglet.gl.GLdouble()
                end_z = pyglet.gl.GLdouble()
                pmat = (pyglet.gl.GLdouble * 16)()
                mvmat = (pyglet.gl.GLdouble * 16)()
                view = (pyglet.gl.GLint * 4)()
                pyglet.gl.glGetDoublev(pyglet.gl.GL_MODELVIEW_MATRIX, mvmat)
                pyglet.gl.glGetDoublev(pyglet.gl.GL_PROJECTION_MATRIX, pmat)
                pyglet.gl.glGetIntegerv(pyglet.gl.GL_VIEWPORT, view)

                pyglet.gl.gluProject(self.models_data[i]['pos_2D'][0], self.models_data[i]['pos_2D'][1],
                                     self.background.zpos, mvmat, pmat, view, near_x,
                                     near_y, near_z)
                pyglet.gl.gluUnProject(near_x, near_y, 0, mvmat, pmat, view, start_x,
                                       start_y, start_z)
                pyglet.gl.gluUnProject(near_x, near_y, 1, mvmat, pmat, view, end_x,
                                       end_y, end_z)
                t = (self.models_data[i]['pos_3D'][2] - start_z.value) / (end_z.value - start_z.value)
                self.models_data[i]['pos_3D'][0] = start_x.value + (end_x.value - start_x.value) * t
                self.models_data[i]['pos_3D'][1] = start_y.value + (end_y.value - start_y.value) * t
                self.models_data[i]['pos_3D'][1] = self.models_data[i]['pos_3D'][1] + 1.0
            self.first = False

        for i in range(len(self.models_data)):
            pyglet.gl.glPushMatrix()
            transl_3D = self.models_data[i]['pos_3D'] + self.models_data[i]['transformation']['translation']
            pyglet.gl.glTranslatef(transl_3D[0], transl_3D[1], transl_3D[2])
            rot_3D = [
                self.models_data[i]['rotation']['pitch'] + self.models_data[i]['transformation']['rotation']['pitch'],
                self.models_data[i]['rotation']['yaw'] + self.models_data[i]['transformation']['rotation']['yaw']]
            pyglet.gl.glRotatef(rot_3D[0], 1.0, 0.0, 0.0)
            pyglet.gl.glRotatef(rot_3D[1], 0.0, 1.0, 0.0)
            visualization.draw(self.models_data[i]['mesh'])
            pmat = (pyglet.gl.GLdouble * 16)()
            mvmat = (pyglet.gl.GLdouble * 16)()
            view = (pyglet.gl.GLint * 4)()
            pyglet.gl.glGetDoublev(pyglet.gl.GL_MODELVIEW_MATRIX, mvmat)
            pyglet.gl.glGetDoublev(pyglet.gl.GL_PROJECTION_MATRIX, pmat)
            pyglet.gl.glGetIntegerv(pyglet.gl.GL_VIEWPORT, view)
            joints_2D = {}
            bbox2D = []
            for key in self.models_data[i]['joints_3D']:
                win_x = pyglet.gl.GLdouble()
                win_y = pyglet.gl.GLdouble()
                win_z = pyglet.gl.GLdouble()
                pyglet.gl.gluProject(self.models_data[i]['joints_3D'][key][0], self.models_data[i]['joints_3D'][key][1],
                                     self.models_data[i]['joints_3D'][key][2], mvmat, pmat, view, win_x, win_y, win_z)
                joints_2D[key] = np.asarray([win_x.value, self.camera_size[1] - win_y.value]).astype(np.int)
            self.models_data[i]['joints_2D'] = joints_2D
            bbox_2D_c = np.zeros((len(self.models_data[i]['box_3D']), 3))
            for j in self.models_data[i]['box_3D']:
                win_x = pyglet.gl.GLdouble()
                win_y = pyglet.gl.GLdouble()
                win_z = pyglet.gl.GLdouble()
                c3D = self.models_data[i]['box_3D'][j]
                pyglet.gl.gluProject(c3D[0], c3D[1], c3D[2], mvmat, pmat, view, win_x, win_y, win_z)
                bbox_2D_c[j] = np.asarray([win_x.value, self.camera_size[1] - win_y.value, win_z.value]).astype(np.int)
            bbox2D.append(np.array([np.min(bbox_2D_c[:, 0]), np.min(bbox_2D_c[:, 1])]))
            bbox2D.append(np.array([np.max(bbox_2D_c[:, 0]), np.max(bbox_2D_c[:, 1])]))
            self.models_data[i]['box_2D'] = np.asarray(bbox2D)
            pyglet.gl.glPopMatrix()

    def update(self, dt):
        if self.shot_ok > 1:
            pyglet.image.get_buffer_manager().get_color_buffer().save('./tmp.png')
            if self.export_data:
                self.write_data()
            self.close()

    def get_data(self):
        models_data = []
        img_data = cv2.imread('./tmp.png')
        os.remove('./tmp.png')
        for i in range(len(self.models_data)):
            bbox = np.array([self.models_data[i]['box_2D'][0][0], self.models_data[i]['box_2D'][0][1],
                             self.models_data[i]['box_2D'][1][0] - self.models_data[i]['box_2D'][0][0],
                             self.models_data[i]['box_2D'][1][1] - self.models_data[i]['box_2D'][0][1]])
            kps = []
            for key in self.models_data[i]['joints_2D']:
                kp = {
                    "name": key,
                    "coords": np.array(
                        [self.models_data[i]['joints_2D'][key][0], self.models_data[i]['joints_2D'][key][1]])
                }
                kps.append(kp)
            model_data = {
                "name": self.models_data[i]['filename'],
                "id": self.models_data[i]['id'],
                "box": bbox,
                "kps": kps

            }
            models_data.append(model_data)
        annot_data = {
            "img": img_data,
            "back_img_name": self.background_img_fl,
            "back_img_id": self.background_img_id,
            "model_data": models_data
        }
        return annot_data

    def write_data(self):
        images_dir = os.path.join(self.data_out_dir, 'images')
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)
        annot_dir = os.path.join(self.data_out_dir, 'labels')
        if not os.path.exists(annot_dir):
            os.mkdir(annot_dir)
        f_lab = os.path.join(annot_dir, '00' + str(int(self.csv_name.replace('data_', '')) + 600000 - 1) + '.csv')
        f_img = os.path.join(images_dir, '00' + str(int(self.csv_name.replace('data_', '')) + 600000 - 1) + '.png')
        pyglet.image.get_buffer_manager().get_color_buffer().save(f_img)
        with open(f_lab, 'w') as csv_lab:
            annot_spec = 'back_img_name,back_img_id,model_name,model_id,bb_x,bb_y,bb_w,bb_h'
            for key in self.models_data[0]['joints_2D']:
                for xy in ['x', 'y']:
                    annot_spec = annot_spec + ',' + key + '_' + xy
            annot_spec = annot_spec + '\n'
            csv_lab.write(annot_spec)
            for i in range(len(self.models_data)):
                annot_data = self.background_img_fl + ',' + str(self.background_img_id) + ',' + self.models_data[i][
                    'filename'] + ',' + str(self.models_data[i]['id']) + ',' + str(
                    self.models_data[i]['box_2D'][0][0]) + ',' + str(self.models_data[i]['box_2D'][0][1]) + ',' + str(
                    self.models_data[i]['box_2D'][1][0] - self.models_data[i]['box_2D'][0][0]) + ',' + str(
                    self.models_data[i]['box_2D'][1][1] - self.models_data[i]['box_2D'][0][1])
                for key in self.models_data[i]['joints_2D']:
                    annot_data = annot_data + ',' + str(self.models_data[i]['joints_2D'][key][0]) + ',' + str(
                        self.models_data[i]['joints_2D'][key][1])
                annot_data = annot_data + '\n'
                csv_lab.write(annot_data)

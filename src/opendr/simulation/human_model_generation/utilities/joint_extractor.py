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

import pyglet
import numpy as np
import sklearn.preprocessing


class Joint_extractor:
    def __init__(self, num_of_joints=18):
        self.num_of_joints = num_of_joints
        self.start_points = []
        self.end_points = []
        for j in range(18):
            self.start_points.append([])
            self.end_points.append([])

    def compute_rays(self, cv_kps, image_width, image_height):
        pmat = (pyglet.gl.GLdouble * 16)()
        mvmat = (pyglet.gl.GLdouble * 16)()
        view = (pyglet.gl.GLint * 4)()
        pyglet.gl.glGetDoublev(pyglet.gl.GL_MODELVIEW_MATRIX, mvmat)
        pyglet.gl.glGetDoublev(pyglet.gl.GL_PROJECTION_MATRIX, pmat)
        pyglet.gl.glGetIntegerv(pyglet.gl.GL_VIEWPORT, view)

        if cv_kps.size != 0:
            for i, cv_kp in enumerate(cv_kps):
                if cv_kp[0] != -1 and cv_kp[0] != -1:
                    start_x = pyglet.gl.GLdouble()
                    start_y = pyglet.gl.GLdouble()
                    start_z = pyglet.gl.GLdouble()
                    end_x = pyglet.gl.GLdouble()
                    end_y = pyglet.gl.GLdouble()
                    end_z = pyglet.gl.GLdouble()
                    pyglet.gl.gluUnProject(cv_kp[0], image_height - cv_kp[1], 0, mvmat, pmat, view, start_x,
                                           start_y, start_z)
                    pyglet.gl.gluUnProject(cv_kp[0], image_height - cv_kp[1], 1, mvmat, pmat, view, end_x, end_y,
                                           end_z)
                    self.start_points[i].append(np.asarray([start_x.value, start_y.value, start_z.value]))
                    self.end_points[i].append(np.asarray([end_x.value, end_y.value, end_z.value]))

    @property
    def compute_3D_positions(self):
        for i in range(self.num_of_joints):
            if len(self.start_points[i]) == 0 or len(self.end_points[i]) == 0:
                print("Failed to estimate the position of the joints...")
                return [[], []]
        points_3D = []
        dists_3D = []
        inds_sorted = None
        for i in range(self.num_of_joints):
            d = 100
            first_time = True
            while d > 0.05:
                if first_time:
                    s = np.asarray(self.start_points[i])
                    e = np.asarray(self.end_points[i])
                else:
                    s = s[inds_sorted[:-1]]
                    e = e[inds_sorted[:-1]]
                v = e - s
                ni = sklearn.preprocessing.normalize(v, norm="l2")
                nx = ni[:, 0]
                ny = ni[:, 1]
                nz = ni[:, 2]
                sxx = np.sum(nx * nx - 1)
                syy = np.sum(ny * ny - 1)
                szz = np.sum(nz * nz - 1)
                sxy = np.sum(nx * ny)
                sxz = np.sum(nx * nz)
                syz = np.sum(ny * nz)
                S = np.asarray([np.asarray([sxx, sxy, sxz]), np.asarray([sxy, syy, syz]), np.asarray([sxz, syz, szz])])
                cx = np.sum(s[:, 0] * (nx * nx - 1) + s[:, 1] * (nx * ny) + s[:, 2] * (nx * nz))
                cy = np.sum(s[:, 0] * (nx * ny) + s[:, 1] * (ny * ny - 1) + s[:, 2] * (ny * nz))
                cz = np.sum(s[:, 0] * (nx * nz) + s[:, 1] * (ny * nz) + s[:, 2] * (nz * nz - 1))
                C = np.asarray([cx, cy, cz])
                p_intersect = np.linalg.inv(np.asarray(S)).dot(C)
                N = s.shape[0]
                distances = np.zeros(N, dtype=np.float32)
                for j in range(N):
                    ui = ((p_intersect - s[j, :]).dot(np.transpose(v[j, :]))) / (v[j, :].dot(v[j, :]))
                    distances[j] = np.linalg.norm(p_intersect - s[j, :] - ui * v[j, :])
                    # for i=1:N %http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html:
                    # distances(i) = norm(cross(p_intersect-PA(i,:),p_intersect-PB(i,:))) / norm(Si(i,:));
                inds_sorted = np.argsort(distances)
                d = distances[inds_sorted[-1]]
                first_time = False
            points_3D.append(p_intersect)
            dists_3D.append(distances)
        points_3D = np.asarray(points_3D, dtype=np.float32)
        dists_3D = np.asarray(dists_3D, dtype=object)
        return points_3D, dists_3D

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

import os
if os.getenv('DISPLAY') is not None:
    from opendr.simulation.human_model_generation.utilities.visualizer import Visualizer


class Model_3D:
    def __init__(self, verts, faces, vert_colors=None):
        self.verts = verts
        self.faces = faces
        self.use_vert_color = False
        if vert_colors is not None:
            self.use_vert_color = True
            self.vert_colors = vert_colors

    def get_vertices(self):
        return self.verts

    def get_faces(self):
        return self.faces

    def save_obj_mesh(self, mesh_path):
        file = open(mesh_path, 'w')
        if self.use_vert_color:
            for idx, v in enumerate(self.verts):
                c = self.vert_colors[idx]
                file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
            for f in self.faces:
                f_plus = f + 1
                file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
        else:
            for v in self.verts:
                file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
            for f in self.faces:
                f_plus = f + 1
                file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
            file.close()

    def get_img_views(self, rotations=None, human_pose_3D=None, plot_kps=False):
        if os.getenv('DISPLAY') is None:
            raise OSError('Renderings of the model can\'t be generated without '
                          'a display...')
        if rotations is None:
            raise ValueError('List of rotations is empty...')
        if human_pose_3D is not None:
            visualizer = Visualizer(out_path='./', mesh=self, pose=human_pose_3D, plot_kps=plot_kps)
        else:
            visualizer = Visualizer(out_path='./', mesh=self)
        return visualizer.infer(rotations=rotations)

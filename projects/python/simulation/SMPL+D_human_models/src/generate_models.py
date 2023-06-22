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

import os
import numpy as np
from mathutils import Vector, Euler
import bpy
import pickle
import mathutils
from shutil import copyfile
import bmesh
import cv2
import tqdm
import io
from contextlib import redirect_stdout


def deselect():
    for o in bpy.data.objects.values():
        o.select_set(False)
    bpy.context.view_layer.objects.active = None


class Fbx_exporter:
    def __init__(self, dir_smpl, dir_model, dir_out):
        self.dir_in = dir_model
        self.dir_smpl = dir_smpl
        self.kintree = {
            -1: (-1, 'root'),
            0: (-1, 'Pelvis'),
            1: (0, 'L_Hip'),
            2: (0, 'R_Hip'),
            3: (0, 'Spine1'),
            4: (1, 'L_Knee'),
            5: (2, 'R_Knee'),
            6: (3, 'Spine2'),
            7: (4, 'L_Ankle'),
            8: (5, 'R_Ankle'),
            9: (6, 'Spine3'),
            10: (7, 'L_Foot'),
            11: (8, 'R_Foot'),
            12: (9, 'Neck'),
            13: (9, 'L_Collar'),
            14: (9, 'R_Collar'),
            15: (12, 'Head'),
            16: (13, 'L_Shoulder'),
            17: (14, 'R_Shoulder'),
            18: (16, 'L_Elbow'),
            19: (17, 'R_Elbow'),
            20: (18, 'L_Wrist'),
            21: (19, 'R_Wrist'),
            22: (20, 'L_Hand'),
            23: (21, 'R_Hand')
        }
        self.n_bones = 24
        self.res = (500, 500)
        self.model_params = {"betas": np.load(os.path.join(dir_model, 'betas.npy')),
                             "gender": str(np.load(os.path.join(dir_model, 'gender.npy'))),
                             "displacements": np.load(os.path.join(dir_model, 'displacements.npy')),
                             "texture": os.path.join(dir_model, 'texture.jpg'),
                             "uv_colored": cv2.imread(os.path.join(self.dir_smpl, 'mask.png'))}

        self.model_params["displacements"] = cv2.resize(self.model_params["displacements"], self.res)
        self.model_params["uv_colored"] = cv2.resize(self.model_params["uv_colored"], self.res)

        self.obname = f'{str(self.model_params["gender"])[0]}_avg'
        self.shape = self.model_params['betas'][:10] / 5.0
        self.arm_ob = None
        self.ob = None
        self.dir_out = dir_out
        if not os.path.isdir(self.dir_out):
            os.mkdir(self.dir_out)
        self.dir_out = os.path.join(self.dir_out, self.model_params['gender'])
        if not os.path.isdir(self.dir_out):
            os.mkdir(self.dir_out)
        self.dir_out = os.path.join(self.dir_out, self.dir_in.split('/')[-1])
        if not os.path.isdir(self.dir_out):
            os.mkdir(self.dir_out)

    def start(self):
        for o in bpy.context.scene.objects:
            if o.type == 'MESH':
                o.select_set(True)
            else:
                o.select_set(False)
        bpy.ops.object.delete()
        bpy.ops.import_scene.fbx(
            filepath=os.path.join(self.dir_smpl,
                                  f'SMPL_{str(self.model_params["gender"])[0]}_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx'),
            axis_forward='Z', axis_up='Y', global_scale=100.0)
        self.ob = bpy.data.objects[self.obname]
        self.ob.data.use_auto_smooth = False

        self.ob.data.shape_keys.animation_data_clear()
        self.arm_ob = bpy.data.objects['Armature']
        self.arm_ob.rotation_euler = Euler((np.radians(0), 0, 0), 'ZYX')
        self.arm_ob.animation_data_clear()

        self.ob.select_set(True)
        bpy.context.view_layer.objects.active = self.ob
        for k in self.ob.data.shape_keys.key_blocks.keys():

            if k != "Basis":
                self.ob.data.shape_keys.key_blocks[k].slider_min = -5
                self.ob.data.shape_keys.key_blocks[k].slider_max = 5
                self.ob.data.shape_keys.key_blocks[k].value = 0

        self.arm_ob.select_set(True)
        bpy.context.view_layer.objects.active = self.arm_ob
        deselect()
        self.ob.select_set(True)
        bpy.context.view_layer.objects.active = self.ob
        self.apply_shape(self.shape)
        bpy.ops.object.select_all(action='DESELECT')
        ob = bpy.context.scene.objects['Armature']
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        ob = bpy.context.scene.objects[self.obname]
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        self.ob.shape_key_add(name="keys_applied", from_mix=True)
        for k in self.ob.data.shape_keys.key_blocks.keys():
            bpy.context.object.active_shape_key_index = 0
            bpy.ops.object.shape_key_remove()
        ob = bpy.context.scene.objects[self.obname]
        for num, m in list(enumerate(ob.material_slots)):
            if m.material:
                m.material.name = "material"
                m.material.use_nodes = True
                bsdf = m.material.node_tree.nodes["Principled BSDF"]
                texImage = m.material.node_tree.nodes.new('ShaderNodeTexImage')
                tex_filename = os.path.join(self.dir_out, 'texture.jpg')
                copyfile(self.model_params["texture"], tex_filename)
                texImage.image = bpy.data.images.load(filepath=tex_filename)
                m.material.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
                ob = bpy.context.view_layer.objects.active

                if ob.data.materials:
                    ob.data.materials[0] = m.material

        disp_norms = self.model_params["displacements"][:, :, 0]
        disp_eul = self.model_params["displacements"][:, :, 1:4]

        with open(os.path.join(self.dir_smpl, 'uv_indices.pkl'), 'rb') as f:
            nrml_ids = pickle.load(f, encoding='latin1')
        with open(os.path.join(self.dir_smpl, 'uv_coords.pkl'), 'rb') as f:
            uv_coords = pickle.load(f, encoding='latin1')

        nrmls = []
        verts_original = []
        scene = bpy.context.scene
        for obj in scene.objects:
            if obj.type == 'MESH':
                bpy.context.view_layer.objects.active = obj
                me = bpy.context.object.data
                bm = bmesh.new()
                bm.from_mesh(me)
                for face in bpy.context.object.data.polygons:
                    f = np.array([0, 0, 0])
                    cnt = 0
                    for idx in face.vertices:
                        f[cnt] = idx
                        cnt = cnt + 1
                for vert in bm.verts:
                    verts_original.append([vert.co.x, vert.co.y, vert.co.z])
                    nrmls.append(np.array([vert.normal.x, vert.normal.y, vert.normal.z]))
                bm.to_mesh(me)
                me.update()
                bm.free()

        disps_final = []
        for i in range(len(ob.data.vertices)):
            disp = Vector((0, 0, 0))
            cnt = 0
            for j in range(len(nrml_ids[i])):
                vt = uv_coords[nrml_ids[i][j]]
                dnorm = disp_norms[int((1 - vt[1]) * disp_norms.shape[0]), int(vt[0] * disp_norms.shape[1])]
                deul = disp_eul[int((1 - vt[1]) * disp_norms.shape[0]), int(vt[0] * disp_norms.shape[1])]
                deul = mathutils.Euler((deul[2], deul[1], deul[0]), 'ZYX')
                disp_inst = Vector(nrmls[i])
                disp_inst.rotate(deul)
                if dnorm > 0:
                    disp = disp + disp_inst * dnorm
                    cnt = cnt + 1
            if cnt > 0:
                disp = disp / cnt
            disps_final.append(disp)

        for obj in scene.objects:
            if obj.type == 'MESH':
                bpy.context.view_layer.objects.active = obj
                me = bpy.context.object.data
                bm = bmesh.new()
                bm.from_mesh(me)
                bm.verts.ensure_lookup_table()
                for i in range(len(bm.verts)):
                    bm.verts[i].co = Vector(verts_original[i]) + disps_final[i] * 100.0

                bm.to_mesh(me)
                for material in me.materials:
                    material.name = 'material'
                me.update()
                bm.free()

        object_types = {'MESH', 'ARMATURE'}
        bpy.ops.export_scene.fbx(filepath=os.path.join(self.dir_out, self.dir_in.split('/')[-1] + '.fbx'),
                                 add_leaf_bones=False,
                                 global_scale=0.01, use_selection=True, object_types=object_types, bake_anim=False)

        for o in bpy.context.scene.objects:
            o.select_set(True)
        objs = bpy.data.objects
        objs.remove(objs['Armature'], do_unlink=True)
        for o in bpy.context.scene.objects:
            o.data.materials.pop(index=0)
            o.data.materials.clear()
        objs.remove(objs[str(self.model_params["gender"])[0] + '_avg'], do_unlink=True)

    def get_bname(self, i, obname='f_avg'):
        return obname + '_' + self.kintree[i][1]

    def apply_shape(self, shape):
        for ibshape, shape_elem in enumerate(shape):
            sign = 'pos'
            if shape_elem < 0:
                sign = 'neg'
            self.ob.data.shape_keys.key_blocks['Shape%03d_' % ibshape + sign].value = abs(shape_elem) * 0.2


def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


if __name__ == '__main__':

    dir_in = './human_models'
    dir_out = './fbx_models'
    dir_smpl = './model'
    dir_models_in_m = [os.path.join(dir_in, 'male', x) for x in next(os.walk(os.path.join(dir_in, 'male')))[1]]
    dir_models_in_f = [os.path.join(dir_in, 'female', x) for x in next(os.walk(os.path.join(dir_in, 'female')))[1]]
    dir_models_in = dir_models_in_f + dir_models_in_m
    pbar = tqdm.tqdm(total=len(dir_models_in))

    for m in range(len(dir_models_in)):
        for myCol in bpy.data.collections:
            obs = [o for o in myCol.objects if o.users == 1]
            while obs:
                bpy.data.objects.remove(obs.pop())
            bpy.data.collections.remove(myCol)

        myCol = bpy.data.collections.new("Collection")
        bpy.context.scene.collection.children.link(myCol)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            fbx_exporter = Fbx_exporter(dir_smpl, dir_models_in[m], dir_out)
            fbx_exporter.start()

        pbar.update(1)
    pbar.close()
    bpy.ops.wm.quit_blender()

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
import numpy as np
from mathutils import Matrix, Vector, Euler
import bpy
import pickle
import mathutils
from shutil import copyfile
import bmesh
import cv2
from numpy import linalg
import math
from scipy.spatial.transform import Rotation as R


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
        self.model_params = {"betas": np.load(os.path.join(dir_model, 'betas.npy')),
                             "gender": str(np.load(os.path.join(dir_model, 'gender.npy'))),
                             "displacements": np.load(os.path.join(dir_model, 'displacements.npy')),
                             "texture": os.path.join(dir_model, 'texture.png'),
                             "uv_colored": cv2.imread(os.path.join(self.dir_smpl, 'mask.png'))}

        self.model_params["displacements"] = cv2.resize(self.model_params["displacements"], (256, 256))
        self.model_params["uv_colored"] = cv2.resize(self.model_params["uv_colored"], (256, 256))

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
            axis_forward='Z', axis_up='Y', global_scale=100.0
        )
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
        self.deselect()
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
                tex_filename = os.path.join(self.dir_out, 'texture.png')
                copyfile(self.model_params["texture"], tex_filename)
                texImage.image = bpy.data.images.load(filepath=tex_filename)
                m.material.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
                ob = bpy.context.view_layer.objects.active

                if ob.data.materials:
                    ob.data.materials[0] = m.material
                else:
                    ob.data.materials.append(m.material)

        disp_norms = self.model_params["displacements"][:, :, 0]
        # cv2.imwrite(os.path.join(self.dir_out,'text.jpg'),disp_norms*255)
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
                i = 0
                for face in bpy.context.object.data.polygons:
                    f = np.array([0, 0, 0])
                    cnt = 0
                    for idx in face.vertices:
                        f[cnt] = idx
                        cnt = cnt + 1
                for vert in bm.verts:
                    verts_original.append([vert.co.x, vert.co.y, vert.co.z])
                    nrmls.append(np.array([vert.normal.x, vert.normal.y, vert.normal.z]))
                    i = i + 1
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

        verts_smooth = []
        for obj in scene.objects:
            if obj.type == 'MESH':
                bpy.context.view_layer.objects.active = obj
                # bpy.ops.object.mode_set(mode='EDIT')
                # bpy.ops.mesh.select_all(action='TOGGLE')
                # bpy.ops.mesh.normals_make_consistent(inside=False)
                me = bpy.context.object.data
                bm = bmesh.new()
                bm.from_mesh(me)
                bm.verts.ensure_lookup_table()
                for i in range(len(bm.verts)):
                    bm.verts[i].co = Vector(verts_original[i]) + disps_final[i]
                # bpy.ops.object.mode_set(mode='OBJECT')
                # bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=1.75, use_axis_x=True, use_axis_y=True, use_axis_z=True)
                bmesh.ops.smooth_laplacian_vert(bm, verts=bm.verts, lambda_factor=0.01, lambda_border=0.01, use_x=True,
                                                use_y=True, use_z=True, preserve_volume=True)

                for vert in bm.verts:
                    verts_smooth.append([vert.co.x, vert.co.y, vert.co.z])
                bm.to_mesh(me)
                me.update()
                bm.free()

        displacement_map_smoothed = np.zeros([disp_norms.shape[0], disp_norms.shape[1], 4])
        displacement_map_smoothed[:, :, 0] = disp_norms
        displacement_map_smoothed[:, :, 1:4] = disp_eul

        for i in range(len(verts_smooth)):
            d_vt = np.asarray(verts_smooth[i]) - np.asarray(verts_original[i])
            dnorm_new = linalg.norm(d_vt)
            if dnorm_new > 0.0:
                d_vt_normalized = d_vt / float(dnorm_new)
                axis = np.cross(nrmls[i], d_vt_normalized)
                axis = axis / linalg.norm(axis)
                angle = math.acos(max(min(np.dot(nrmls[i], d_vt_normalized), 1), -1))
                rot_mat = R.from_matrix(rotation_matrix_from_vectors(nrmls[i], d_vt_normalized))
                deul_new = rot_mat.as_euler('ZYX', degrees=False)
            for j in range(len(nrml_ids[i])):
                vt = uv_coords[nrml_ids[i][j]]
                if dnorm_new > 0 and np.dot(nrmls[i], d_vt_normalized) > 0 and linalg.norm(disps_final[i]) > 0:
                    an = [0.8, 0.2]
                    for s1 in range(-1, 2):
                        for s2 in range(-1, 2):
                            ii = min(max(0, int((1 - vt[1]) * disp_norms.shape[0]) + s1), disp_norms.shape[0] - 1)
                            jj = min(max(0, int(vt[0] * disp_norms.shape[1]) + s2), disp_norms.shape[1] - 1)
                    displacement_map_smoothed[ii, jj, 1:4] = an[0] * deul_new + an[1] * disp_eul[ii, jj]
                    displacement_map_smoothed[ii, jj, 0] = an[0] * dnorm_new + an[1] * disp_norms[ii, jj]
        vts = []
        for i in range(len(ob.data.vertices)):
            for j in range(len(nrml_ids[i])):
                vts.append(np.array([int((1 - uv_coords[nrml_ids[i][j]][1]) * disp_norms.shape[0]),
                                     int(uv_coords[nrml_ids[i][j]][0] * disp_norms.shape[1])]))
        vts = np.asarray(vts)
        times = 1
        displacement_map_smoothed_new = np.zeros(self.model_params["displacements"].shape)

        for t in range(times):
            for i in range(displacement_map_smoothed.shape[0]):
                print(str(i) + ' ' + str(displacement_map_smoothed.shape[0]))
                for j in range(displacement_map_smoothed.shape[1]):
                    vt_current = np.asarray([i, j])
                    clr = self.model_params["uv_colored"][i, j, :]
                    if not np.equal(clr, np.array([0, 0, 0])).all():
                        vts_now = vts.copy()
                        inds = np.where(
                            self.model_params["uv_colored"][vts[:, 0], vts[:, 1], 0] == clr[0]) and np.where(
                            self.model_params["uv_colored"][vts[:, 0], vts[:, 1], 1] == clr[1]) and np.where(
                            self.model_params["uv_colored"][vts[:, 0], vts[:, 1], 2] == clr[2])

                        if inds[0].shape[0] > 2:
                            vts_now = vts_now[inds[0], :]
                            dists = (vts_now[:, 1] - vt_current[1]) * (vts_now[:, 1] - vt_current[1]) + (
                                    vts_now[:, 0] - vt_current[0]) * (
                                            vts_now[:, 0] - vt_current[0])
                            ids_sorted = np.argsort(dists)
                            factors_all = dists[ids_sorted[0]] + dists[ids_sorted[1]] + dists[ids_sorted[2]]
                            if self.model_params["displacements"][i][j][0] > 0:
                               displacement_map_smoothed_new[i, j, :] = (1 - dists[ids_sorted[0]] / factors_all) * \
                                         displacement_map_smoothed[
                                         vts_now[ids_sorted[0], :][0],
                                         vts_now[ids_sorted[0], :][1], :] + \
                                         (1 - dists[ids_sorted[1]] / factors_all) * \
                                         displacement_map_smoothed[
                                         vts_now[ids_sorted[1], :][0],
                                         vts_now[ids_sorted[1], :][1], :] + \
                                         (1 - dists[ids_sorted[2]] / factors_all) * \
                                         displacement_map_smoothed[
                                         vts_now[ids_sorted[2], :][0],
                                         vts_now[ids_sorted[2], :][1], :]
                                displacement_map_smoothed_new[i, j, :] = displacement_map_smoothed_new[i, j, :] / 2.0
        displacement_map_smoothed = displacement_map_smoothed_new

        # for i in range(displacement_map_smoothed.shape[0]):
        #    for j in range(displacement_map_smoothed.shape[0]):
        #        if np.array_equal(self.model_params["eyes_d"][i, j, :], np.array([255, 255, 255, 255])):
        #            displacement_map_smoothed[i, j, :] = np.array([0, 0, 0, 0])

        text_img = cv2.imread(self.model_params['texture'], cv2.IMREAD_UNCHANGED)
        text_img = cv2.resize(text_img, (displacement_map_smoothed.shape[0], displacement_map_smoothed.shape[1]))
        # cv2.imshow('aa', displacement_map_smoothed)
        # cv2.imshow('bb', text_img)
        # cv2.waitKey(0)
        '''
        displacement_map_smoothed_n = np.zeros(displacement_map_smoothed.shape)
        for i in range(text_img.shape[0]):
            for j in range(text_img.shape[1]):
                cnt = 0
                cnt_v = [0, 0, 0, 0]
                for si in range(-1, 1, 1):
                    for sj in range(-1, 1, 1):
                        ii = min(max(0, i + si), text_img.shape[0] - 1)
                        jj = min(max(0, j + sj), text_img.shape[1] - 1)
                        if not math.isnan(displacement_map_smoothed[ii, jj, 0]):
                            cnt_v[0] = cnt_v[0] + float(displacement_map_smoothed[ii, jj, 0])
                            cnt_v[1] = cnt_v[1] + float(displacement_map_smoothed[ii, jj, 1])
                            cnt_v[2] = cnt_v[2] + float(displacement_map_smoothed[ii, jj, 2])
                            cnt_v[3] = cnt_v[3] + float(displacement_map_smoothed[ii, jj, 3])
                            cnt = cnt + 1
                if cnt > 0:
                    displacement_map_smoothed_n[i, j, 0] = cnt_v[0]/cnt
                    displacement_map_smoothed_n[i, j, 1] = cnt_v[1]/cnt
                    displacement_map_smoothed_n[i, j, 2] = cnt_v[2]/cnt
                    displacement_map_smoothed_n[i, j, 3] = cnt_v[3]/cnt
                else:
                    displacement_map_smoothed_n[i, j, 0] = 0
                    displacement_map_smoothed_n[i, j, 1] = 0
                    displacement_map_smoothed_n[i, j, 2] = 0
                    displacement_map_smoothed_n[i, j, 3] = 0
        displacement_map_smoothed = displacement_map_smoothed_n
        '''

        clr = []
        clr.append(np.array(text_img[int(0.37 * text_img.shape[0]), int(0.37 * text_img.shape[1]), :]).astype(int))
        clr.append(np.array(text_img[int(0.31 * text_img.shape[0]), int(0.31 * text_img.shape[1]), :]).astype(int))
        clr_extra = np.array(text_img[int(0.56 * text_img.shape[0]), int(0.66 * text_img.shape[1]), :]).astype(int)
        if (clr[0] + clr[1])[0] / 2.0 + 30 > clr_extra[0] and (clr[0] + clr[1])[1] / 2.0 + 30 > clr_extra[1] and \
                (clr[0] + clr[1])[2] / 2.0 + 30 > clr_extra[2] and ((clr[0] + clr[1])[0] / 2.0 - 30) < clr_extra[
            0] and ((clr[0] + clr[1])[1] / 2.0 - 30) < clr_extra[1] and ((clr[0] + clr[1])[2] / 2.0 - 30) < clr_extra[
            2]:
            clr.append(clr_extra)
        clr_extra = np.array(text_img[int(0.62 * text_img.shape[0]), int(0.94 * text_img.shape[1]), :]).astype(int)
        if (clr[0] + clr[1])[0] / 2.0 + 30 > clr_extra[0] and (clr[0] + clr[1])[1] / 2.0 + 30 > clr_extra[1] and \
                (clr[0] + clr[1])[2] / 2.0 + 30 > clr_extra[2] and ((clr[0] + clr[1])[0] / 2.0 - 30) < clr_extra[
            0] and ((clr[0] + clr[1])[1] / 2.0 - 30) < clr_extra[1] and ((clr[0] + clr[1])[2] / 2.0 - 30) < clr_extra[
            2]:
            clr.append(clr_extra)
        thres = 15
        for i in range(text_img.shape[0]):
            for j in range(text_img.shape[1]):
                if self.model_params["uv_colored"][i, j, 0] == 0:  # or math.isnan(displacement_map_smoothed[i, j, 0]):
                    displacement_map_smoothed[i, j, 0] = 0
                    displacement_map_smoothed[i, j, 1] = 0
                    displacement_map_smoothed[i, j, 2] = 0
                    displacement_map_smoothed[i, j, 3] = 0
                else:
                    for si in range(-1, 2, 1):
                        for sj in range(-1, 2, 1):
                            ii = min(max(0, i + si), text_img.shape[0] - 1)
                            jj = min(max(0, j + sj), text_img.shape[1] - 1)
                            for c in range(len(clr)):
                                if text_img[ii, jj, 0] > (clr[c][0] - thres) and text_img[ii, jj, 0] < (
                                        clr[c][0] + thres) and text_img[ii, jj, 1] > (clr[c][1] - thres) and text_img[
                                    ii, jj, 1] < (clr[c][1] + thres) and text_img[ii, jj, 2] > (clr[c][2] - thres) and \
                                        text_img[ii, jj, 2] < (clr[c][2] + thres):
                                    displacement_map_smoothed[ii, jj, 0] = 0
                                    displacement_map_smoothed[ii, jj, 1] = 0
                                    displacement_map_smoothed[ii, jj, 2] = 0
                                    displacement_map_smoothed[ii, jj, 3] = 0

        '''
        for i in range(text_img.shape[0]):
            for j in range(text_img.shape[1]):
                cnt = 0
                for si in range(-1, 2, 1):
                    for sj in range(-1, 2, 1):
                        ii = min(max(0, i + si), text_img.shape[0] - 1)
                        jj = min(max(0, j + sj), text_img.shape[1] - 1)
                        if displacement_map_smoothed[ii, jj, 0] > 0:
                            cnt = cnt + 1
                if cnt > 0:
                    displacement_map_smoothed_n[ii, jj] = 255
        displacement_map_smoothed = displacement_map_smoothed_n
        '''

        cv2.imwrite(os.path.join(self.dir_out, self.dir_in.split('/')[-1] + '_c.png'), disp_eul * 255)
        cv2.imwrite(os.path.join(self.dir_out, self.dir_in.split('/')[-1] + '_d.png'),
                    displacement_map_smoothed[:, :, 1:4] * 255)
        cv2.imwrite(os.path.join(self.dir_out, self.dir_in.split('/')[-1] + '_e.png'), disp_norms * 255)
        cv2.imwrite(os.path.join(self.dir_out, self.dir_in.split('/')[-1] + '_f.png'),
                    displacement_map_smoothed[:, :, 0] * 255)
        disps_final2 = []
        for i in range(len(verts_smooth)):
            disp = Vector((0, 0, 0))
            cnt = 0
            for j in range(len(nrml_ids[i])):
                vt = uv_coords[nrml_ids[i][j]]
                dnorm = displacement_map_smoothed[int((1 - vt[1]) * displacement_map_smoothed.shape[0]), int(
                    vt[0] * displacement_map_smoothed.shape[1])][0]
                deul = displacement_map_smoothed[int((1 - vt[1]) * displacement_map_smoothed.shape[0]), int(
                    vt[0] * displacement_map_smoothed.shape[1])][1:4]
                deul = mathutils.Euler((deul[2], deul[1], deul[0]), 'ZYX')
                disp_inst = Vector(nrmls[i])
                disp_inst.rotate(deul)
                if dnorm > 0:
                    disp = disp + disp_inst * dnorm
                    cnt = cnt + 1
                    # print(disp)
            if cnt > 0:
                disp = disp / cnt
            disps_final2.append(disp)

        for obj in scene.objects:
            if obj.type == 'MESH':
                bpy.context.view_layer.objects.active = obj
                # bpy.ops.object.mode_set(mode='EDIT')
                # bpy.ops.mesh.select_all(action='TOGGLE')
                # bpy.ops.mesh.normals_make_consistent(inside=False)
                me = bpy.context.object.data
                bm = bmesh.new()
                bm.from_mesh(me)
                bm.verts.ensure_lookup_table()
                for i in range(len(bm.verts)):
                    bm.verts[i].co = Vector(verts_original[i]) + disps_final2[i] * 100.0
                bm.to_mesh(me)
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
        objs.remove(objs[str(self.model_params["gender"])[0] + '_avg'], do_unlink=True)

    def deselect(self):
        for o in bpy.data.objects.values():
            o.select_set(False)
        bpy.context.view_layer.objects.active = None

    def get_bname(self, i, obname='f_avg'):
        return obname + '_' + self.kintree[i][1]

    def apply_shape(self, shape):
        for ibshape, shape_elem in enumerate(shape):
            sign = 'pos'
            if shape_elem < 0:
                sign = 'neg'
            self.ob.data.shape_keys.key_blocks['Shape%03d_' % ibshape + sign].value = abs(shape_elem) * 0.2


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


if __name__ == '__main__':

    dir_in = './human_data'
    dir_out = './fbx_models'
    dir_smpl = './model'
    dir_models_in = [os.path.join(dir_in, x) for x in next(os.walk(dir_in))[1]]

    for i in range(len(dir_models_in)):
        for myCol in bpy.data.collections:
            obs = [o for o in myCol.objects if o.users == 1]
            while obs:
                bpy.data.objects.remove(obs.pop())
            bpy.data.collections.remove(myCol)

        myCol = bpy.data.collections.new("Collection")
        bpy.context.scene.collection.children.link(myCol)
        fbx_exporter = Fbx_exporter(dir_smpl, dir_models_in[i], dir_out)
        fbx_exporter.start()

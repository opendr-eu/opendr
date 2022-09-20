import os.path as osp
import numpy as np
import torch
import scipy.io as sio
import pickle
from ...data import curve
import skimage.transform as trans
from math import cos, sin, atan2, asin
import neural_renderer as nr


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


def P2sRt(P):
    ''' decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    '''
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def matrix2angle(R):
    ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    '''
    # assert(isRotationMatrix(R))

    if R[2, 0] != 1 and R[2, 0] != -1:
        x = -asin(max(-1, min(R[2, 0], 1)))
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])

    return [x, y, z]


def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: yaw.
        y: pitch.
        z: roll.
    Returns:
        R: 3x3. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x, y, z = angles[0], angles[1], angles[2]
    y, x, z = angles[0], angles[1], angles[2]

    # x
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), -sin(x)],
                   [0, sin(x), cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, sin(y)],
                   [0, 1, 0],
                   [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0],
                   [sin(z), cos(z), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    return R.astype(np.float32)


class Render(object):
    def __init__(self, opt):
        self.opt = opt
        self.render_size = opt.crop_size
        print(self.render_size, opt.crop_size)
        self.d = './algorithm/DDFA/train.configs'
        w_shp = _load(osp.join(self.d, 'w_shp_sim.npy'))
        w_exp = _load(osp.join(self.d, 'w_exp_sim.npy'))  # simplified version
        u_shp = _load(osp.join(self.d, 'u_shp.npy'))
        u_exp = _load(osp.join(self.d, 'u_exp.npy'))
        self.keypoints = _load(osp.join(self.d, 'keypoints_sim.npy'))
        self.pose_noise = getattr(opt, 'pose_noise', False)
        self.large_pose = getattr(opt, 'large_pose', False)
        u = u_shp + u_exp
        tri = sio.loadmat('./algorithm/DDFA/visualize/tri.mat')['tri']  # 3 * 53215
        faces_np = np.expand_dims(tri.T, axis=0).astype(np.int32) - 1

        self.std_size = 120

        opt.gpu_ids = 0

        self.current_gpu = opt.gpu_ids
        with torch.cuda.device(self.current_gpu):
            self.faces = torch.from_numpy(faces_np).cuda()
            self.renderer = nr.Renderer(camera_mode='look', image_size=self.render_size, perspective=False,
                                        light_intensity_directional=0, light_intensity_ambient=1)
            self.u_cuda = torch.from_numpy(u.astype(np.float32)).cuda()
            self.w_shp_cuda = torch.from_numpy(w_shp.astype(np.float32)).cuda()
            self.w_exp_cuda = torch.from_numpy(w_exp.astype(np.float32)).cuda()

    def random_p(self, s, angle):

        if np.random.randint(0, 2) == 0:
            angle[0] += np.random.uniform(-0.965, -0.342, 1)[0]
            # angle[1] += np.random.uniform(-0.1, 0.1, 1)[0]
        else:
            angle[0] += np.random.uniform(0.342, 0.965, 1)[0]
            # angle[1] += np.random.uniform(-0.1, 0.1, 1)[0]
        angle[0] = max(-1.2, min(angle[0], 1.2))
        random_2 = np.random.uniform(-0.5, 0.5, 1)[0]
        angle[1] += random_2
        angle[1] = max(-1.0, min(angle[1], 1.0))
        p = angle2matrix(angle) * s
        return p

    def assign_large(self, s, angle):
        if np.random.randint(0, 2) == 0:
            angle[0] = np.random.uniform(-1.05, -0.95, 1)[0]
            # angle[1] += np.random.uniform(-0.1, 0.1, 1)[0]
        else:
            angle[0] = np.random.uniform(1.05, 0.95, 1)[0]
            # angle[1] += np.random.uniform(-0.1, 0.1, 1)[0]
        angle[0] = max(-1.2, min(angle[0], 1.2))
        random_2 = np.random.uniform(-0.5, 0.5, 1)[0]
        angle[1] += random_2
        angle[1] = max(-1.0, min(angle[1], 1.0))
        p = angle2matrix(angle) * s
        return p

    def _parse_param(self, param, pose_noise=False, frontal=True,
                     large_pose=False, yaw_pose=None, pitch_pose=None):
        """Work for both numpy and tensor"""
        p_ = param[:12].reshape(3, -1)
        p = p_[:, :3]
        s, R, t3d = P2sRt(p_)
        angle = matrix2angle(R)
        original_angle = angle[0]
        if yaw_pose is not None or pitch_pose is not None:
            # angle[0] = yaw_pose if yaw_pose is not None
            if yaw_pose is not None:
                angle[0] = yaw_pose
                # flag = -1 if angle[0] < 0 else 1
                # angle[0] = flag * abs(yaw_pose)
            if pitch_pose is not None:
                angle[1] = pitch_pose
                # flag = -1 if angle[1] < 0 else 1
                # angle[1] = flag * abs(pitch_pose)
            # elif angle[1] < 0:
                # angle[1] = 0
            p = angle2matrix(angle) * s
        else:
            if frontal:
                angle[0] = 0
                if angle[1] < 0:
                    angle[1] = 0
                p = angle2matrix(angle) * s
            if pose_noise:
                if frontal:
                    if np.random.randint(0, 5):
                        p = self.random_p(s, angle)
                else:
                    p = self.random_p(s, angle)
            elif large_pose:
                if frontal:
                    if np.random.randint(0, 5):
                        p = self.assign_large(s, angle)
                else:
                    p = self.assign_large(s, angle)

        offset = p_[:, -1].reshape(3, 1)
        alpha_shp = param[12:52].reshape(-1, 1)
        alpha_exp = param[52:-4].reshape(-1, 1)
        box = param[-4:]
        return p, offset, alpha_shp, alpha_exp, box, original_angle

    def affine_align(self, landmark=None, **kwargs):
        # M = None
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)
        src = src * 290 / 112
        src[:, 0] += 50
        src[:, 1] += 60
        src = src / 400 * self.render_size
        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M2 = tform.params[0:2, :]
        with torch.cuda.device(self.current_gpu):
            M2 = torch.from_numpy(M2).float().cuda()
        return M2

    def texture_vertices_to_faces(self, tex_input, faces):
        # tex_input: (B, N, 2, 2, 2, C)
        # faces: (faceN, 3)
        faces = faces.long()
        tex_out = tex_input[:, faces[0, :, 0], :] + tex_input[:, faces[0, :, 1], :] + tex_input[:, faces[0, :, 2], :]
        return tex_out / 3.0

    def compute_tri_normal(self, vertex, tri):
        # Unit normals to the faces
        # vertex : 3xvertex_num
        # tri : 3xtri_num

        vt1_indices, vt2_indices, vt3_indices = torch.split(tri.t(), split_size_or_sections=1, dim=1)

        vt1 = vertex[vt1_indices[:, 0], :]
        vt2 = vertex[vt2_indices[:, 0], :]
        vt3 = vertex[vt3_indices[:, 0], :]

        normalf = (vt2 - vt1).cross(vt3 - vt1)
        normalf = torch.nn.functional.normalize(normalf, dim=1, p=2)

        return normalf

    def vertices_rescale(self, v, roi_bbox):
        vertices = v.clone()
        sx, sy, ex, ey = roi_bbox
        scale_x = (ex - sx) / 120
        scale_y = (ey - sy) / 120
        vertices[0, :] = vertices[0, :] * scale_x + sx
        vertices[1, :] = vertices[1, :] * scale_y + sy
        s = (scale_x + scale_y) / 2
        vertices[2, :] *= s
        return vertices

    def get_five_points(self, vertices):
        indexs = [4150, 11744, 8191, 5650, 10922]
        five_points = np.zeros((5, 2))
        for i, idx in enumerate(indexs):
            five_points[i, :] = vertices[0:2, idx]
        return five_points

    def get_68_points(self, vertices):
        vertixes = vertices.T.flatten()
        vertice_68 = vertixes[self.keypoints].reshape(-1, 3)
        vertice_68 = vertice_68.astype(np.int)
        return vertice_68

    def torch_get_68_points(self, vertices):
        vertixes = vertices.transpose(1, 2).contiguous()
        vertixes = vertixes.view(vertixes.size(0), -1)
        vertice_68 = vertixes[:, self.keypoints].reshape(vertices.size(0), -1, 3)
        return vertice_68

    def transform_vertices(self, M, vertices):
        # M = M.float()
        v_size = vertices.size()
        # M = torch.Tensor(M).cuda()
        with torch.cuda.device(self.current_gpu):
            M = M.float().cuda()
        R = M[:, :2]
        t = M[:, 2]
        vertices2 = vertices.clone()
        vertices2 = vertices2.float()
        vertices2[:2, :] = R.mm(vertices2[:2, :]) + t.repeat(v_size[1], 1).t()
        return vertices2

    def generate_vertices_and_rescale_to_img(self, param, pose_noise=False,
                                             mean_shp=False, mean_exp=False, frontal=True, large_pose=False,
                                             yaw_pose=None, pitch_pose=None):
        p, offset, alpha_shp, alpha_exp, roi_bbox, original_angle = self._parse_param(param, pose_noise=pose_noise,
                                                                                      frontal=frontal,
                                                                                      large_pose=large_pose,
                                                                                      yaw_pose=yaw_pose,
                                                                                      pitch_pose=pitch_pose)
        if mean_shp:
            alpha_shp.fill(0.0)
        if mean_exp:
            alpha_exp.fill(0.0)
        with torch.cuda.device(self.current_gpu):
            p = torch.from_numpy(p.astype(np.float32)).cuda()
            alpha_shp = torch.from_numpy(alpha_shp.astype(np.float32)).cuda()
            alpha_exp = torch.from_numpy(alpha_exp.astype(np.float32)).cuda()
            offset = torch.from_numpy(offset.astype(np.float32)).cuda()

        vertices = p.matmul(
            (self.u_cuda + self.w_shp_cuda.matmul(alpha_shp) + self.w_exp_cuda.matmul(alpha_exp)).view(-1,
                                                                                                       3).t()) + offset

        vertices[1, :] = self.std_size + 1 - vertices[1, :]
        vertices = self.vertices_rescale(vertices, roi_bbox)

        return vertices, original_angle

    def flip_normalize_vertices(self, vertices):
        # flip and normalize vertices
        vertices[1, :] = self.render_size - vertices[1, :] - 1
        vertices[:2, :] = vertices[:2, :] / (self.render_size / 2.0) - 1.0
        vertices[2, :] = (vertices[2, :] - vertices[2, :].min()) / (vertices[2, :].max() - vertices[2, :].min()) * 2 - 1
        vertices[2, :] = -1.0 * vertices[2, :]
        vertices = vertices.t().unsqueeze(0)
        return vertices

    def get_render_from_vertices(self, img_ori, vertices_in_ori_img):
        c, h, w = img_ori.size()
        img_ori = img_ori.clone().permute(1, 2, 0)
        # random_num = np.random.randint(30000, 50000)
        #         vertices_in_ori_img[:,30000:50000] = vertices_in_ori_img[:,30000:50000] * 1.02 - 3
        # vertices_in_ori_img[:, 20000:random_num] = vertices_in_ori_img[:, 20000:random_num] * np.random.uniform(1.01,
        #                                                                          1.02) - np.random.uniform(0.5, 1.5)

        textures = img_ori[vertices_in_ori_img[1, :].round().clamp(0, h - 1).long(), vertices_in_ori_img[0, :].round().clamp(
            0, w - 1).long(), :]

        N = textures.shape[0]
        with torch.cuda.device(self.current_gpu):
            textures = textures.cuda().view(1, N, 1, 1, 1, 3)
        textures = textures.expand(1, N, 2, 2, 2, 3)
        textures = textures.float()
        tex_a = self.texture_vertices_to_faces(textures, self.faces)

        return tex_a

    def _forward(self, param_file, img_ori, M=None,
                 pose_noise=True, mean_exp=False, mean_shp=False, align=True, frontal=True,
                 large_pose=False, yaw_pose=None, pitch_pose=None):
        '''
        img_ori: rgb image, normalized within 0-1, h * w * 3
        return: render image, bgr
        '''
        param = np.fromfile(param_file, sep=' ')

        vertices, original_angle = self.generate_vertices_and_rescale_to_img(param, pose_noise=pose_noise,
                                                                             mean_shp=mean_shp, mean_exp=mean_exp,
                                                                             frontal=frontal,
                                                                             large_pose=large_pose, yaw_pose=yaw_pose,
                                                                             pitch_pose=pitch_pose)

        if not (pose_noise or mean_exp or mean_exp or frontal):
            print('pose_noise')
            print(not pose_noise or mean_exp or mean_exp or frontal)
            if M is not None:
                vertices = self.transform_vertices(M, vertices)
            else:
                five_points = self.get_five_points(vertices.cpu().numpy())
                M = self.affine_align(five_points)
                vertices = self.transform_vertices(M, vertices)
            vertices_in_ori_img = vertices.clone()
            align_vertices = vertices.clone()
        else:
            vertices_in_ori_img, _ = self.generate_vertices_and_rescale_to_img(param, pose_noise=False,
                                                                               mean_shp=False, mean_exp=False,
                                                                               frontal=False, large_pose=False)
            if M is not None:
                vertices_in_ori_img = self.transform_vertices(M, vertices_in_ori_img)
            else:
                five_points = self.get_five_points(vertices_in_ori_img.cpu().numpy())
                M = self.affine_align(five_points)
                vertices_in_ori_img = self.transform_vertices(M, vertices_in_ori_img)

            five_points = self.get_five_points(vertices.cpu().numpy())
            M_0 = self.affine_align(five_points)

            # if np.random.randint(0, 4) < 1:
            if align:
                vertices = self.transform_vertices(M_0, vertices)
                align_vertices = vertices.clone()
            else:
                align_vertices = vertices.clone()
                align_vertices = self.transform_vertices(M_0, align_vertices)
                vertices = self.transform_vertices(M, vertices)

        with torch.cuda.device(self.current_gpu):
            img_ori = img_ori.cuda()
        c, h, w = img_ori.size()
        assert h == w

        vertices_in_ori_img[:2, :] = vertices_in_ori_img[:2, :] / self.render_size * h
        # original image size is 400 * 400 * 3

        # original image size is 400 * 400 * 3
        vertices_out = vertices.clone()
        tex_a = self.get_render_from_vertices(img_ori, vertices_in_ori_img)
        vertices = self.flip_normalize_vertices(vertices)
        vertices_in_ori_img[:2, :] = vertices_in_ori_img[:2, :] / h * self.render_size
        return tex_a, vertices, vertices_out, vertices_in_ori_img, align_vertices, original_angle

    def rotate_render(self, params, images, M=None, with_BG=False, pose_noise=False, large_pose=False,
                      align=True, frontal=True, erode=True, grey_background=False, avg_BG=True,
                      yaw_pose=None, pitch_pose=None):

        bz, c, w, h = images.size()
        pose_noise = self.pose_noise
        large_pose = self.large_pose
        face_size = self.faces.size()
        self.faces_use = self.faces.expand(bz, face_size[1], face_size[2])

        # get render color vertices and normal vertices information, get original texs
        vertices = []
        vertices_out = []
        vertices_in_ori_img = []
        vertices_aligned_normal = []
        vertices_aligned_out = []
        vertices_ori_normal = []
        texs = []
        original_angles = torch.zeros(bz)
        with torch.no_grad():
            for n in range(bz):
                tex_a, vertice, vertice_out, vertice_in_ori_img, align_vertice, original_angle \
                    = self._forward(params[n], images[n], M[n],
                                    pose_noise=pose_noise, align=align, frontal=frontal,
                                    large_pose=large_pose, yaw_pose=yaw_pose, pitch_pose=pitch_pose)
                vertices.append(vertice)
                vertices_out.append(vertice_out)
                vertices_in_ori_img.append(vertice_in_ori_img.clone())
                vertice2 = self.flip_normalize_vertices(vertice_in_ori_img.clone())
                vertices_ori_normal.append(vertice2)
                vertices_aligned_out.append(align_vertice)
                align_vertice_normal = self.flip_normalize_vertices(align_vertice.clone())
                vertices_aligned_normal.append(align_vertice_normal.clone())
                texs.append(tex_a)
                original_angles[n] = original_angle

            vertices = torch.cat(vertices, 0)
            vertices_aligned_normal = torch.cat(vertices_aligned_normal, 0)
            vertices_ori_normal = torch.cat(vertices_ori_normal, 0)

            vertices_in_ori_img = torch.stack(vertices_in_ori_img, 0)
            vertices_aligned_out = torch.stack(vertices_aligned_out, 0)

            texs = torch.cat(texs, 0)
            texs_old = texs.clone()

            # erode the original mask and render again
            rendered_images_erode = None
            if erode:

                with torch.cuda.device(self.current_gpu):
                    rendered_images, depths, masks, = self.renderer(vertices_ori_normal, self.faces_use, texs)
                    # rendered_images: batch * 3 * h * w, masks: batch * h * w
                masks_erode = self.generate_erode_mask(masks, kernal_size=self.opt.erode_kernel)
                rendered_images = rendered_images.cpu()
                Rd_a = rendered_images.clone()
                if grey_background:
                    rendered_images_erode = masks_erode * rendered_images
                else:

                    inv_masks_erode = (torch.ones_like(masks_erode) - (masks_erode)).float()
                    if avg_BG:
                        contentsum = torch.sum(torch.sum(masks_erode * rendered_images, 3), 2)
                        sumsum = torch.sum(torch.sum(masks_erode, 3), 2)
                        contentsum[contentsum == 0] = 0.5
                        sumsum[sumsum == 0] = 1
                        masked_sum = contentsum / sumsum
                        masked_BG = masked_sum.unsqueeze(2).unsqueeze(3).expand(rendered_images.size())
                    else:
                        masked_BG = 0.5
                    rendered_images_erode = masks_erode * rendered_images + inv_masks_erode * masked_BG

                texs_a_crop = []
                for n in range(bz):
                    tex_a_crop = self.get_render_from_vertices(rendered_images_erode[n], vertices_in_ori_img[n])
                    texs_a_crop.append(tex_a_crop)
                texs = torch.cat(texs_a_crop, 0)

            # render face to rotated pose
            with torch.cuda.device(self.current_gpu):
                rendered_images, depths, masks, = self.renderer(vertices, self.faces_use, texs)

            # add mask to rotated
            masks_erode = self.generate_erode_mask(masks, kernal_size=5)
            inv_masks_erode = (torch.ones_like(masks_erode) - masks_erode).float()
            rendered_images = rendered_images.cpu()
            if with_BG:
                images = torch.nn.functional.interpolate(images, size=(self.render_size))
                rendered_images = masks_erode * rendered_images + inv_masks_erode * images  # 3 * h * w
            else:
                if grey_background:
                    if np.random.randint(0, 4):
                        rendered_images = masks_erode * rendered_images
                else:
                    if avg_BG:
                        contentsum = torch.sum(torch.sum(masks_erode * rendered_images, 3), 2)
                        sumsum = torch.sum(torch.sum(masks_erode, 3), 2)
                        contentsum[contentsum == 0] = 0.5
                        sumsum[sumsum == 0] = 1
                        masked_sum = contentsum / sumsum
                        masked_BG = masked_sum.unsqueeze(2).unsqueeze(3).expand(rendered_images.size())
                    else:
                        masked_BG = 0.5
                    rendered_images = masks_erode * rendered_images + inv_masks_erode * masked_BG

            # get rendered face vertices
            texs_b = []
            for n in range(bz):
                tex_b = self.get_render_from_vertices(rendered_images[n], vertices_out[n])
                texs_b.append(tex_b)
            texs_b = torch.cat(texs_b, 0)

            # render back
            with torch.cuda.device(self.current_gpu):
                rendered_images_rotate, depths1, masks1, = self.renderer(vertices_ori_normal, self.faces_use, texs_b)
                # rendered_images: batch * 3 * h * w, masks: batch * h * w
                rendered_images_rotate_artifacts, depths1, masks1, = self.renderer(vertices_aligned_normal, self.faces_use,
                                                                                   texs_old)
                # rendered_images: batch * 3 * h * w, masks: batch * h * w
                rendered_images_double, depths2, masks2, = self.renderer(vertices_aligned_normal, self.faces_use, texs_b)
                # rendered_images: batch * 3 * h * w, masks: batch * h * w

            masks2 = masks2.unsqueeze(1)
            inv_masks2 = (torch.ones_like(masks2) - masks2).float().cpu()
            # BG = inv_masks2 * images
            if grey_background:
                masks1 = masks1.unsqueeze(1)

                inv_masks1 = (torch.ones_like(masks1) - masks1).float()

                rendered_images_rotate = (inv_masks1 * 0.5 + rendered_images_rotate).clamp(0, 1)
                rendered_images_double = (inv_masks2 * 0.5 + rendered_images_double).clamp(0, 1)

        artifacts = rendered_images_rotate_artifacts
        return rendered_images_rotate, rendered_images_double, self.torch_get_68_points(
            vertices_in_ori_img), self.torch_get_68_points(
            vertices_aligned_out), rendered_images_erode, original_angles, Rd_a, artifacts

    def generate_erode_mask(self, masks, kernal_size=5):
        masks = masks.unsqueeze(1)
        masks = masks.cpu()
        with torch.no_grad():
            Conv = torch.nn.Conv2d(1, 1, kernal_size, padding=(kernal_size // 2), bias=False)
            Conv.weight.fill_(1 / (kernal_size * kernal_size))
            masks2 = Conv(masks)
        random_start1 = np.random.randint(50, 100)
        masks[:, :, random_start1:self.render_size - 10, :] = masks2[:, :, random_start1:self.render_size - 10, :]
        masks = (masks > np.random.uniform(0.8, 0.99)).float()
        return masks

    def get_seg_map(self, vertices, no_guassian=False, size=256):
        landmarks = self.torch_get_68_points(vertices)
        landmarks = landmarks[:, :, :2].cpu().numpy().astype(np.float)
        all_heatmap = []
        all_orig_heatmap = []
        for i in range(landmarks.shape[0]):
            heatmap = curve.points_to_heatmap_68points(landmarks[i], 13, size, self.opt.heatmap_size)
            heatmap2 = curve.combine_map(heatmap, no_guassian=no_guassian)
            all_heatmap.append(heatmap2)
            all_orig_heatmap.append(heatmap)
        all_heatmap = np.stack(all_heatmap, axis=0)
        all_orig_heatmap = np.stack(all_orig_heatmap, axis=0)
        all_heatmap = torch.from_numpy(all_heatmap.astype(np.float32)).cuda()
        all_orig_heatmap = torch.from_numpy(all_orig_heatmap.astype(np.float32)).cuda()
        all_orig_heatmap = all_orig_heatmap.permute(0, 3, 1, 2)
        all_orig_heatmap[all_orig_heatmap > 0] = 1.0
        return all_heatmap, all_orig_heatmap

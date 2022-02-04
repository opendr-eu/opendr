import torch
import pickle
import numpy as np
from .render import osp, Render


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


class TestRender(Render):

    def __init__(self, opt):
        super(TestRender, self).__init__(opt)
        self.keypoints_106 = _load(osp.join(self.d, '106_index.npy'))

    def torch_get_106_points(self, vertices):
        vertixes = vertices.transpose(1, 2).contiguous()
        vertixes = vertixes.view(vertixes.size(0), -1)
        vertice_106 = vertixes[:, self.keypoints_106].reshape(vertices.size(0), -1, 3)
        return vertice_106

    def rotate_render(self, params, images, M=None, with_BG=False, pose_noise=False, large_pose=False,
                      align=True, frontal=True, erode=True, grey_background=False, avg_BG=True,
                      yaw_pose=None, pitch_pose=None):

        bz, c, w, h = images.size()

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
                                    yaw_pose=yaw_pose, pitch_pose=pitch_pose)
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

            # erode the original mask and render again
            rendered_images_erode = None
            if erode:
                with torch.cuda.device(self.current_gpu):
                    rendered_images, depths, masks, = self.renderer(vertices_ori_normal, self.faces_use,
                                                                    texs)
                    # rendered_images: batch * 3 * h * w, masks: batch * h * w
                masks_erode = self.generate_erode_mask(masks, kernal_size=15)
                rendered_images = rendered_images.cpu()
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
            with torch.no_grad():
                with torch.cuda.device(self.current_gpu):
                    rendered_images, depths, masks, = self.renderer(vertices, self.faces_use, texs)

        return rendered_images, self.torch_get_68_points(
            vertices_aligned_out), original_angles, self.torch_get_106_points(vertices_aligned_out)

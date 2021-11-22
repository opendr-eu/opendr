import torch
from . import networks
from .rotatespade_model import RotateSPADEModel


class TestModel(RotateSPADEModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super(TestModel, self).__init__(opt)

    def forward(self, data, mode):
        if mode == 'single':
            real_image = data['image']
            rotated_landmarks = data['rotated_landmarks']
            original_angles = data['original_angles']
            self.rotated_seg, rotated_seg_all = \
                self.get_seg_map(rotated_landmarks, self.opt.no_gaussian_landmark, self.opt.crop_size, original_angles)
            rotated_mesh = data['rotated_mesh']
            if self.opt.label_mask:
                rotated_mesh = (rotated_mesh + rotated_seg_all[:, 4].unsqueeze(1) + rotated_seg_all[:, 0].unsqueeze(1))
                rotated_mesh[rotated_mesh >= 1] = 0
            with torch.no_grad():
                fake_rotate, _ = self.generate_fake(rotated_mesh, real_image, self.rotated_seg)

            return fake_rotate

        else:
            real_image = data['image']

            orig_landmarks = data['orig_landmarks']
            rotated_landmarks = data['rotated_landmarks']
            orig_seg, orig_seg_all = self.get_seg_map(orig_landmarks, self.opt.no_gaussian_landmark, self.opt.crop_size)
            rotated_seg, rotated_seg_all = self.get_seg_map(rotated_landmarks, self.opt.no_gaussian_landmark,
                                                            self.opt.crop_size)

            input_semantics = data['mesh']
            rotated_mesh = data['rotated_mesh']
            # BG = data['BG']

            if self.opt.label_mask:
                input_semantics = (input_semantics + orig_seg_all[:, 4].unsqueeze(1) + orig_seg_all[:, 0].unsqueeze(1))
                rotated_mesh = (rotated_mesh + rotated_seg_all[:, 4].unsqueeze(1) + rotated_seg_all[:, 0].unsqueeze(1))
                input_semantics[input_semantics >= 1] = 0
                rotated_mesh[rotated_mesh >= 1] = 0

            with torch.no_grad():
                if self.opt.label_mask:
                    rotated_mesh = (
                            rotated_mesh + rotated_seg_all[:, 4].unsqueeze(1) + rotated_seg_all[:, 0].unsqueeze(1))
                    rotated_mesh[rotated_mesh >= 1] = 0
                fake_image, _ = self.generate_fake(input_semantics, real_image, self.orig_seg)
                fake_rotate, _ = self.generate_fake(rotated_mesh, real_image, self.rotated_seg)

            return fake_image, fake_rotate

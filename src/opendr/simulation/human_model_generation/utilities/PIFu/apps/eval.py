from opendr.simulation.human_model_generation.utilities.PIFu.lib.options import BaseOptions
from opendr.simulation.human_model_generation.utilities.PIFu.lib.train_util import gen_mesh_color
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image as PIL_image
# get options
opt = BaseOptions().parse()


class Evaluator:
    def __init__(self, opt, netG, netC, cuda, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.cuda = cuda
        self.netG = netG
        self.netC = netC

    def load_image(self, image, mask):
        # Name
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib = torch.Tensor(projection_matrix).float()
        # Mask
        mask = transforms.ToTensor()(mask).float()
        # image
        image = self.to_tensor(PIL_image.fromarray(image[:, :, ::-1]))
        image = mask.expand_as(image) * image
        return {
            'img': image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
            # save_path = '%s/%s.obj' % (opt.results_path, data['name'])
            return gen_mesh_color(opt, self.netG, self.netC, self.cuda, data, use_octree=use_octree)

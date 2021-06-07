from projects.simulation.human_model_generation.PIFu.lib.options import BaseOptions
from projects.simulation.human_model_generation.PIFu.lib.train_util import gen_mesh_color
from projects.simulation.human_model_generation.PIFu.lib.model import ResBlkPIFuNet, HGPIFuNet
import torchvision.transforms as transforms
import torch
import numpy as np
# get options
opt = BaseOptions().parse()


class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        if opt.cuda and torch.cuda.is_available():
            cuda = torch.device('cuda:%d' % opt.gpu_id)
        else:
            cuda = torch.device('cpu')

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=cuda)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        else:
            netC = None

        # os.makedirs(opt.results_path, exist_ok=True)
        # os.makedirs('%s' % (opt.results_path), exist_ok=True)

        # opt_log = os.path.join(opt.results_path, 'opt.txt')
        # with open(opt_log, 'w') as outfile:
        #    outfile.write(json.dumps(vars(opt), indent=2))

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
        image = self.to_tensor(image)
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

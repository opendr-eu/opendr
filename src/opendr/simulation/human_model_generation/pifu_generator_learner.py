from opendr.simulation.human_model_generation.utilities.PIFu.lib.options import BaseOptions
from opendr.simulation.human_model_generation.utilities.PIFu.apps.eval import Evaluator
from opendr.simulation.human_model_generation.utilities.PIFu.apps.crop_img import process_imgs
from opendr.simulation.human_model_generation.utilities.model_3D import Model_3D
from opendr.simulation.human_model_generation.utilities.visualizer import Visualizer
import os
from opendr.simulation.human_model_generation.utilities.studio import Studio
import wget
from os import path
from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.simulation.human_model_generation.utilities.PIFu.lib.model import ResBlkPIFuNet, HGPIFuNet
from opendr.engine.constants import OPENDR_SERVER_URL
import torch
import json
from urllib.request import urlretrieve


class PIFuGeneratorLearner(Learner):
    def __init__(self, device='cpu'):
        super().__init__()
        self.opt = BaseOptions().parse()
        checkpoint_dir = os.path.join(os.path.split(__file__)[0], 'utilities', 'PIFu', 'checkpoints')
        net_G_path = os.path.join(checkpoint_dir, 'net_G')
        net_C_path = os.path.join(checkpoint_dir, 'net_C')
        self.download(checkpoint_dir)
        if device == 'cuda':
            self.opt.cuda = True
        self.opt.load_netG_checkpoint_path = net_G_path
        self.opt.load_netC_checkpoint_path = net_C_path
        # Network configuration
        self.opt.batch_size = 1
        self.opt.mlp_dim = [257, 1024, 512, 256, 128, 1]
        self.opt.mlp_dim_color = [513, 1024, 512, 256, 128, 3]
        self.opt.num_stack = 4
        self.opt.num_hourglass = 2
        self.opt.resolution = 256
        self.opt.hg_down = 'ave_pool'
        self.opt.norm = 'group'
        self.opt.norm_color = 'group'
        self.opt.projection_mode = 'orthogonal'
        # create net
        # set cuda
        if self.opt.cuda and torch.cuda.is_available():
            self.cuda = torch.device('cuda:%d' % self.opt.gpu_id)
        else:
            self.cuda = torch.device('cpu')
        self.netG = HGPIFuNet(self.opt, self.opt.projection_mode).to(device=self.cuda)
        self.netC = ResBlkPIFuNet(self.opt).to(device=self.cuda)
        self.load('./utilities/PIFu/checkpoints')
        self.evaluator = Evaluator(self.opt, self.netG, self.netC, self.cuda)

    def infer(self, imgs_rgb, imgs_msk=None, obj_path=None, extract_pose=False):
        for i in range(len(imgs_rgb)):
            if not isinstance(imgs_rgb[i], Image):
                imgs_rgb[i] = Image(imgs_rgb[i])
            imgs_rgb[i] = imgs_rgb[i].numpy()
        for i in range(len(imgs_msk)):
            if not isinstance(imgs_msk[i], Image):
                imgs_msk[i] = Image(imgs_msk[i])
            imgs_msk[i] = imgs_msk[i].numpy()
        if imgs_msk is None or len(imgs_rgb) != 1 or len(imgs_msk) != 1:
            print('Wrong input...')
            return
        if imgs_rgb[0].size != imgs_msk[0].size:
            print('Images must have the same resolution...')
            return
        if (obj_path is not None) and (not os.path.exists(os.path.dirname(obj_path))):
            print("OBJ cannot be saved in the given directory...")
            return
        try:
            [imgs_rgb[0], imgs_msk[0]] = process_imgs(imgs_rgb[0], imgs_msk[0])
            [verts, faces, colors] = self.evaluator.eval(self.evaluator.load_image(imgs_rgb[0], imgs_msk[0]), use_octree=True)
            model_3D = Model_3D(verts, faces, vert_colors=colors)
            if obj_path is not None:
                model_3D.save_obj_mesh(obj_path)
            if extract_pose:
                studio = Studio()
                studio.infer(model_3D=model_3D)
                human_poses_3D = studio.get_poses()
                return [model_3D, human_poses_3D]
            return model_3D
        except Exception as e:
            print("error:", e.args)

    def load(self, path):
        with open(os.path.join(path, "PIFu_default.json")) as metadata_file:
            metadata = json.load(metadata_file)
            load_netG_checkpoint_path = os.path.join(path, metadata['model_paths'][1])
            load_netC_checkpoint_path = os.path.join(path, metadata['model_paths'][0])
            self.netG.load_state_dict(torch.load(load_netG_checkpoint_path, map_location=self.cuda))
            self.netC.load_state_dict(torch.load(load_netC_checkpoint_path, map_location=self.cuda))
            print("PIFu model is loaded.")

    def optimize(self, **kwargs):
        pass

    def reset(self):
        pass

    def save(self, **kwargs):
        pass

    def eval(self, **kwargs):
        pass

    def fit(self, **kwargs):
        pass

    def get_img_views(self, model_3D, rotations, human_pose_3D=None, plot_kps=False):
        if human_pose_3D is not None:
            visualizer = Visualizer(out_path='./', mesh=model_3D, pose=human_pose_3D, plot_kps=plot_kps)
        else:
            visualizer = Visualizer(out_path='./', mesh=model_3D)
        return visualizer.infer(rotations=rotations)

    def download(self, path=None,
                 url=OPENDR_SERVER_URL + "simulation/human_model_generation/checkpoints/"):
        if path is None:
            path = self.temp_path

        if not os.path.exists(path):
            os.makedirs(path)

        if (not os.path.exists(os.path.join(path, "PIFu_default.json"))) or \
                (not os.path.exists(os.path.join(path, "net_C"))) or \
                (not os.path.exists(os.path.join(path, "net_G"))):
            print("Downloading pretrained model...")
            file_url = os.path.join(url, "PIFu_defaults.json")
            urlretrieve(file_url, os.path.join(path, "PIFu_defaults.json"))
            file_url = os.path.join(url, "netC")
            urlretrieve(file_url, os.path.join(path, "netC"))

            file_url = os.path.join(url, "netG")
            urlretrieve(file_url, os.path.join(path, "netG"))

            print("Pretrained model download complete.")

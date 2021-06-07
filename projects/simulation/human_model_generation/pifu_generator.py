from projects.simulation.human_model_generation.model_generator import ModelGenerator
from projects.simulation.human_model_generation.PIFu.lib.options import BaseOptions
from projects.simulation.human_model_generation.PIFu.apps.eval import Evaluator
from projects.simulation.human_model_generation.PIFu.apps.crop_img import process_imgs
from projects.simulation.human_model_generation.model_3D import Model_3D
from projects.simulation.human_model_generation.visualizer import Visualizer
import os
from projects.simulation.human_model_generation.studio import Studio
import wget
from os import path


class PIFuGenerator(ModelGenerator):
    def __init__(self, device='cpu'):
        super().__init__()
        self.opt = BaseOptions().parse()
        checkpoint_dir = os.path.join(os.path.split(__file__)[0], 'PIFu', 'checkpoints')
        net_G_path = os.path.join(checkpoint_dir, 'net_G')
        net_C_path = os.path.join(checkpoint_dir, 'net_C')
        if not path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        if not path.exists(net_G_path):
            wget.download("https://drive.google.com/uc?export=download&id=1zEmVXG2VHy0MMzngcRshB4D8Sr_oLHsm",
                          out=checkpoint_dir)
        if not path.exists(net_C_path):
            wget.download("https://drive.google.com/uc?export=download&id=1V83B6GDIjYMfHdpg-KcCSAPgHxpafHgd",
                          out=checkpoint_dir)
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
        self.evaluator = Evaluator(self.opt)

    def infer(self, imgs_rgb, imgs_msk=None, obj_path=None, extract_pose=False):
        if imgs_msk is None or len(imgs_rgb) != 1 or len(imgs_msk) != 1:
            print('Wrong input...')
            return
        if imgs_rgb[0].size != imgs_msk[0].size:
            print('Images must have the same resolution...')
            return
        if (obj_path is not None) and (not os.path.exists(os.path.dirname(obj_path))):
            print("OBJ cannot be saved in the given directory...")
            return
        if imgs_msk[0].mode != 'L':
            imgs_msk[0] = imgs_msk[0].convert('L')
        if imgs_rgb[0].mode != 'RGB':
            imgs_rgb[0] = imgs_rgb[0].convert('RGB')
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

    def get_img_views(self, model_3D, rotations, human_pose_3D=None, plot_kps=False):
        if human_pose_3D is not None:
            visualizer = Visualizer(out_path='./', mesh=model_3D, pose=human_pose_3D, plot_kps=plot_kps)
        else:
            visualizer = Visualizer(out_path='./', mesh=model_3D)
        return visualizer.infer(rotations=rotations)

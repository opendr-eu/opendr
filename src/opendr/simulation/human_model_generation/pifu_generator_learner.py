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

from opendr.simulation.human_model_generation.utilities.PIFu.lib.options import BaseOptions
from opendr.simulation.human_model_generation.utilities.PIFu.apps.eval import Evaluator
from opendr.simulation.human_model_generation.utilities.PIFu.apps.crop_img import process_imgs
from opendr.simulation.human_model_generation.utilities.model_3D import Model_3D
import os
from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.engine.constants import OPENDR_SERVER_URL
import torch
import json
from urllib.request import urlretrieve
from opendr.simulation.human_model_generation.utilities.PIFu.pifu_funcs import config_vanilla_parameters, config_nets
if os.getenv('DISPLAY') is not None:
    from opendr.simulation.human_model_generation.utilities.config_utils import config_studio


class PIFuGeneratorLearner(Learner):
    def __init__(self, device='cpu', checkpoint_dir=None):
        super().__init__()
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(os.environ['OPENDR_HOME'], 'src', 'opendr', 'simulation',
                                          'human_model_generation',  'utilities', 'PIFu', 'checkpoints')
        self.opt = BaseOptions().parse()

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.opt.load_netG_checkpoint_path = os.path.join(checkpoint_dir, 'net_G')
        self.opt.load_netC_checkpoint_path = os.path.join(checkpoint_dir, 'net_C')
        self.download(checkpoint_dir)
        self.opt = config_vanilla_parameters(self.opt)

        # set cuda
        if device == 'cuda' and torch.cuda.is_available():
            self.opt.cuda = True
            self.cuda = torch.device('cuda:%d' % self.opt.gpu_id)
        else:
            self.cuda = torch.device('cpu')

        [self.netG, self.netC] = config_nets(self.opt, self.cuda)
        self.load(checkpoint_dir)
        self.evaluator = Evaluator(self.opt, self.netG, self.netC, self.cuda)

    def infer(self, imgs_rgb=None, imgs_msk=None, obj_path=None, extract_pose=False):
        if os.getenv('DISPLAY') is None and extract_pose is True:
            raise ValueError('Pose can\'t be extracted without rendering the generated'
                             'model on a display...')
        for i in range(len(imgs_rgb)):
            if not isinstance(imgs_rgb[i], Image):
                imgs_rgb[i] = Image(imgs_rgb[i])
            imgs_rgb[i] = imgs_rgb[i].convert("channels_last", "bgr")

        for i in range(len(imgs_msk)):
            if not isinstance(imgs_msk[i], Image):
                imgs_msk[i] = Image(imgs_msk[i])
            imgs_msk[i] = imgs_msk[i].convert("channels_last", "bgr")

        if imgs_msk is None or len(imgs_rgb) != 1 or len(imgs_msk) != 1:
            raise ValueError('At the current stage, both the length of the RGB images and the length of the images used as '
                             'masks must be of size equal to 1.')
        if imgs_rgb[0].size != imgs_msk[0].size:
            raise ValueError('Images must have the same resolution...')
        if (obj_path is not None) and (not os.path.exists(os.path.dirname(obj_path))):
            os.mkdir(os.path.dirname(obj_path))
        try:
            [imgs_rgb[0], imgs_msk[0]] = process_imgs(imgs_rgb[0], imgs_msk[0])
            [verts, faces, colors] = self.evaluator.eval(self.evaluator.load_image(imgs_rgb[0], imgs_msk[0]), use_octree=True)
            model_3D = Model_3D(verts, faces, vert_colors=colors)
            if obj_path is not None:
                model_3D.save_obj_mesh(obj_path)
            if extract_pose:
                studio = config_studio()
                studio.infer(model_3D=model_3D)
                human_poses_3D = studio.get_poses()
                return [model_3D, human_poses_3D]
            return model_3D
        except Exception as e:
            raise e

    def load(self, path=None):
        with open(os.path.join(path, "PIFu_default.json")) as metadata_file:
            metadata = json.load(metadata_file)
            load_netG_checkpoint_path = os.path.join(path, metadata['model_paths'][1])
            load_netC_checkpoint_path = os.path.join(path, metadata['model_paths'][0])
            self.netG.load_state_dict(torch.load(load_netG_checkpoint_path, map_location=self.cuda))
            self.netC.load_state_dict(torch.load(load_netC_checkpoint_path, map_location=self.cuda))
            print("PIFu model is loaded.")

    def optimize(self, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def save(self, **kwargs):
        raise NotImplementedError

    def eval(self, **kwargs):
        raise NotImplementedError

    def fit(self, **kwargs):
        raise NotImplementedError

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
            file_url = os.path.join(url, "PIFu_default.json")
            urlretrieve(file_url, os.path.join(path, "PIFu_default.json"))
            file_url = os.path.join(url, "net_C")
            urlretrieve(file_url, os.path.join(path, "net_C"))

            file_url = os.path.join(url, "net_G")
            urlretrieve(file_url, os.path.join(path, "net_G"))

            print("Pretrained model download complete.")

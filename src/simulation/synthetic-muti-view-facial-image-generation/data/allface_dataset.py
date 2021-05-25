import os
import math
import numpy as np
from PIL import Image
import skimage.transform as trans
import cv2
import torch
from data import dataset_info
from data.base_dataset import BaseDataset
import util.util as util

dataset_info = dataset_info()

class AllFaceDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def cv2_loader(self, img_str):
        img_array = np.frombuffer(img_str, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    def fill_list(self, tmp_list):
        length = len(tmp_list)
        if length % self.opt.batchSize != 0:
            end = math.ceil(length / self.opt.batchSize) * self.opt.batchSize
            tmp_list = tmp_list + tmp_list[-1 * (end - length) :]
        return tmp_list

    def initialize(self, opt):
        self.opt = opt
        dataset_num = dataset_info.get_dataset(opt)
        self.prefix = [dataset_info.prefix[num] for num in dataset_num]

        file_list = [dataset_info.file_list[num] for num in dataset_num]

        land_mark_list = [dataset_info.land_mark_list[num] for num in dataset_num]

        self.params_dir = [dataset_info.params_dir[num] for num in dataset_num]

        self.folder_level = [dataset_info.folder_level[num] for num in dataset_num]


        self.num_datasets = len(file_list)
        assert len(land_mark_list) == self.num_datasets, \
        'num of landmk dir should be the num of datasets'

        assert len(self.params_dir) == self.num_datasets, \
        'num of params_dir should be the num of datasets'

        self.dataset_lists = []
        self.landmark_paths = []
        self.sizes = []

        for n in range(self.num_datasets):

            with open(file_list[n]) as f:
                img_lists = f.readlines()
            img_lists = self.fill_list(img_lists)
            self.sizes.append(len(img_lists))
            self.dataset_lists.append(sorted(img_lists))

            with open(land_mark_list[n]) as f:
                landmarks = f.readlines()
                landmarks = self.fill_list(landmarks)
                self.landmark_paths.append(sorted(landmarks))

        self.dataset_size = min(self.sizes)
        self.initialized = False

    def get_landmarks(self, landmark, img_list):

        landmark_split = landmark.strip().split(' ')
        filename1_without_ext = os.path.basename(img_list.strip())
        filename2_without_ext = os.path.basename(landmark_split[0])
        assert (filename1_without_ext == filename2_without_ext), \
            "The image_path %s and params_path %s don't match." % \
            (img_list, landmark_split[0])

        label = landmark_split[1]
        landmarks = landmark_split[2:]
        landmarks = list(map(float, landmarks))
        landmarks_array = np.array(landmarks).reshape(5, 2)
        return landmarks_array, label

    def get_param_file(self, img_list, dataset_num):
        img_name = os.path.splitext(img_list)[0]
        name_split = img_name.split("/")

        folder_level = self.folder_level[dataset_num]
        param_folder = os.path.join(self.params_dir[dataset_num],
                                    "/".join([name_split[i] for i in range(len(name_split) - folder_level, len(name_split))]) + ".txt")
        #     params = np.loadtxt(param_folder)
        return param_folder

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1)[-10:])[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2)[-10:])[0]
        return filename1_without_ext == filename2_without_ext

    def affine_align(self, img, landmark=None, **kwargs):
        M = None
        h, w, c = img.shape
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)
        src = src * 290 / 112
        src[:, 0] += 50
        src[:, 1] += 60
        src = src / 400 * self.opt.crop_size
        dst = landmark
        # dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        warped = cv2.warpAffine(img, M, (self.opt.crop_size, self.opt.crop_size), borderValue=0.0)
        return warped, M

    def __getitem__(self, index):
        # Label Image

        randnum = np.random.randint(sum(self.sizes))
        dataset_num = np.random.randint(self.num_datasets)

        image_path = self.dataset_lists[dataset_num][index].strip()
        image_path = os.path.join(self.prefix[dataset_num], image_path)

        img = cv2.imread(image_path)
        if img is None:
            raise Exception('None Image')

        param_path = self.get_param_file(image_path, dataset_num)

        # img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        M = None
        landmark_path = self.landmark_paths[dataset_num][index].strip()
        landmarks, label = self.get_landmarks(landmark_path, image_path)
        wrapped_img, M = self.affine_align(img, landmarks)
        M = torch.from_numpy(M).float()

        wrapped_img = wrapped_img.transpose(2, 0, 1) / 255.0

        wrapped_img = torch.from_numpy(wrapped_img).float()

        input_dict = {
                      'image': wrapped_img,
                      'param_path': param_path,
                      'M': M,
                      'path': image_path
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

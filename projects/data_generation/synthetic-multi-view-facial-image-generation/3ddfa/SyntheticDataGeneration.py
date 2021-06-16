# Copyright 1996-2020 OpenDR European Project
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

# MIT License
#
# Copyright (c) 2019 Jian Zhao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# !/usr/bin/env python3
# coding: utf-8
from tqdm import tqdm
from shutil import copyfile
import cv2
import os
import main
import inference
import test_multipose
import argparse
from utils.ddfa import str2bool
from opendr.engine.learners import Learner


class MultiviewDataGenerationLearner(Learner):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    def __init__(self):

        self.rootdir = input("Enter the folder with subfolders of images: ")
        self.path = input("Enter the path of folder 3ddfa, e.g.'/home/<user>/Documents/Rotate-and-Render/3ddfa/': ")
        self.key = str(self.path + "/example/Images/")
        self.key1 = str(self.path + "/example/")
        self.key2 = str(self.path + "/results/")

        parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
        parser.add_argument('-f', '--files', nargs='+',
                            help='image files paths fed into network, single or multiple images')
        parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
        parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')
        parser.add_argument('--bbox_init', default='one', type=str,
                            help='one|two: one-step bbox initialization or two-step')
        parser.add_argument('--dump_res', default='true', type=str2bool,
                            help='whether write out the visualization image')
        parser.add_argument('--dump_vertex', default='false', type=str2bool,
                            help='whether write out the dense face vertices to mat')
        parser.add_argument('--dump_ply', default='true', type=str2bool)
        parser.add_argument('--dump_pts', default='true', type=str2bool)
        parser.add_argument('--dump_roi_box', default='false', type=str2bool)
        parser.add_argument('--dump_pose', default='true', type=str2bool)
        parser.add_argument('--dump_depth', default='true', type=str2bool)
        parser.add_argument('--dump_pncc', default='true', type=str2bool)
        parser.add_argument('--dump_paf', default='true', type=str2bool)
        parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
        parser.add_argument('--dump_obj', default='true', type=str2bool)
        parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
        parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                            help='whether use dlib landmark to crop image')
        self.args1 = parser.parse_args()

        parser2 = argparse.ArgumentParser(description='3DDFA inference pipeline')
        parser2.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
        parser2.add_argument('--bbox_init', default='two', type=str,
                             help='one|two: one-step bbox initialization or two-step')
        parser2.add_argument('--dump_2d_img', default='true', type=str2bool, help='whether to save 3d rendered image')
        parser2.add_argument('--dump_param', default='true', type=str2bool, help='whether to save param')
        parser2.add_argument('--dump_lmk', default='true', type=str2bool, help='whether to save landmarks')
        parser2.add_argument('--save_dir', default=self.key2, type=str, help='dir to save result')
        parser2.add_argument('--save_lmk_dir', default='./example', type=str, help='dir to save landmark result')
        parser2.add_argument('--img_list', default='./txt_name_batch.txt', type=str, help='test image list file')

        parser2.add_argument('--rank', default=0, type=int, help='used when parallel run')
        parser2.add_argument('--world_size', default=1, type=int, help='used when parallel run')
        parser2.add_argument('--resume_idx', default=0, type=int)

        self.args2 = parser2.parse_args()

        super(Learner, self).__init__()

    def eval(self):

        # STAGE No1 : detect faces and fitting to 3d mesh by main.py execution
        list_im = []

        print("START")

        a = open("file_list.txt", "w")
        for subdir, dirs, files in os.walk(self.rootdir):
            print(subdir)
            print(dirs)
            current_directory_path = os.path.abspath(subdir)

            print(current_directory_path)
            for file in files:
                name, ext = os.path.splitext(file)
                if ext == ".jpg":
                    current_image_path = os.path.join(current_directory_path, file)
                    print(current_image_path)
                    current_image = cv2.imread(current_image_path)
                    list_im.append(current_image_path)
                    a.write(str(file) + os.linesep)
                    cv2.imwrite(os.path.join(self.key, file), current_image)
            self.args1.files = list_im.copy()
            list_im.clear()
            main.main(self.args1)
        a.close()

        # STAGE No2: Landmarks Output with inference.py execution

        im_list2 = []
        d = open(os.path.join(self.key1, 'realign_lmk'), "w")
        for subdir, dirs, files in os.walk(self.rootdir):
            current_directory_path = os.path.abspath(subdir)
            self.args2.img_prefix = current_directory_path
            self.args2.save_dir = os.path.abspath(self.key2)
            self.args2.save_lmk_dir = os.path.abspath(self.key1)
            if not os.path.exists(self.args2.save_dir):
                os.mkdir(self.args2.save_dir)
            if not os.path.exists(self.args2.save_lmk_dir):
                os.mkdir(self.args2.save_lmk_dir)

            print(current_directory_path)
            list_lfw_batch = './file_list.txt'
            dst = os.path.join(self.args2.save_lmk_dir, "file_list.txt")
            copyfile(list_lfw_batch, dst)
            b = open("txt_name_batch.txt", "w")
            for file in files:

                with open(list_lfw_batch) as f:
                    img_list = [x.strip() for x in f.readlines()]

                    for img_idx, img_fp in enumerate(tqdm(img_list)):
                        if img_fp == str(file):
                            im_list2.append(str(file))
                            b.write(str(file) + os.linesep)
            print(im_list2)
            self.args2.img_list = './txt_name_batch.txt'
            b.close()
            self.args2.dump_lmk = 'true'
            im_list2.clear()
            inference.main(self.args2)
            with open(os.path.join(self.args2.save_lmk_dir, 'realign_lmk_')) as f:
                img_list = [x.strip() for x in f.readlines()]
                for img_idx, img_fp in enumerate(tqdm(img_list)):
                    d.write(img_fp + os.linesep)
        d.close()

        # STAGE No3: Generate Facial Images in specific pitch and yaw angles
        test_multipose.main()


def fit(self):
    # do nothing
    print("do nothing")


def infer(self):
    # do nothing
    print("do nothing")


def load(self):
    # do nothing
    print("do nothing")


def optimize(self):
    # do nothing
    print("do nothing")


def reset(self):
    # do nothing
    print("do nothing")


def save(self):
    # do nothing
    print("do nothing")

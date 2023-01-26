# Copyright 2020-2023 OpenDR European Project
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

# !/usr/bin/env python3.7
# coding: utf-8
from tqdm import tqdm
from shutil import copyfile
import cv2
import os
from .algorithm.DDFA import preprocessing_1
from .algorithm.DDFA import preprocessing_2
from .algorithm.Rotate_and_Render import test_multipose


class MultiviewDataGeneration():

    def __init__(self, args):

        self.path_in = args.path_in
        self.key = str(args.path_3ddfa + "/example/Images/")
        self.key1 = str(args.path_3ddfa + "/example/")
        self.key2 = str(args.path_3ddfa + "/results/")
        self.save_path = args.save_path
        self.val_yaw = args.val_yaw
        self.val_pitch = args.val_pitch
        self.args = args

    def eval(self):

        # STAGE No1 : detect faces and fitting to 3d mesh by main.py execution
        list_im = []

        print("START")

        a = open("file_list.txt", "w")
        for subdir, dirs, files in os.walk(self.path_in):
            current_directory_path = os.path.abspath(subdir)
            for file in files:
                name, ext = os.path.splitext(file)
                if ext == ".jpg":
                    current_image_path = os.path.join(current_directory_path, file)
                    current_image = cv2.imread(current_image_path)
                    list_im.append(current_image_path)
                    a.write(str(file) + os.linesep)
                    cv2.imwrite(os.path.join(self.key, file), current_image)
            self.args.files = list_im.copy()
            list_im.clear()
            preprocessing_1.main(self.args)
        a.close()

        # STAGE No2: Landmarks Output with inference.py execution

        im_list2 = []
        d = open(os.path.join(self.key1, 'realign_lmk'), "w")
        for subdir, dirs, files in os.walk(self.path_in):
            current_directory_path = os.path.abspath(subdir)
            self.args.img_prefix = current_directory_path
            self.args.save_dir = os.path.abspath(self.key2)
            self.args.save_lmk_dir = os.path.abspath(self.key1)
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            if not os.path.exists(self.args.save_lmk_dir):
                os.mkdir(self.args.save_lmk_dir)

            list_lfw_batch = './file_list.txt'
            dst = os.path.join(self.args.save_lmk_dir, "file_list.txt")
            copyfile(list_lfw_batch, dst)
            b = open("txt_name_batch.txt", "w")
            for file in files:

                with open(list_lfw_batch) as f:
                    img_list = [x.strip() for x in f.readlines()]

                    for img_idx, img_fp in enumerate(tqdm(img_list)):
                        if img_fp == str(file):
                            im_list2.append(str(file))
                            b.write(str(file) + os.linesep)
            self.args.img_list = './txt_name_batch.txt'
            b.close()
            self.args.dump_lmk = 'true'
            im_list2.clear()
            preprocessing_2.main(self.args)
            with open(os.path.join(self.args.save_lmk_dir, 'realign_lmk_')) as f:
                img_list = [x.strip() for x in f.readlines()]
                for img_idx, img_fp in enumerate(tqdm(img_list)):
                    d.write(img_fp + os.linesep)
        d.close()

        # STAGE No3: Generate Facial Images in specific pitch and yaw angles
        test_multipose.main(self.save_path, self.val_yaw, self.val_pitch)

    def fit(self):
        raise NotImplementedError()

    def infer(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def optimize(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

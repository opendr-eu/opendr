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
import torch
import matplotlib
import os
import argparse
from SyntheticDataGeneration import MultiviewDataGeneration
from algorithm.DDFA.utils.ddfa import str2bool

matplotlib.use('Agg')
__all__ = ['torch']

if __name__ == '__main__':
    print("\n\n**********************************\nTEST Multiview Data Generation Learner\n"
          "**********************************")
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", default="cuda", type=str, help="choose between cuda or cpu ")
    parser.add_argument("-path_in", default=os.path.join("opendr_internal", "projects",
                                                         "data_generation",
                                                         "",
                                                         "demos", "imgs_input"),
                        type=str, help='Give the path of image folder')
    parser.add_argument('-path_3ddfa', default=os.path.join("opendr_internal", "projects",
                                                            "data_generation",
                                                            "",
                                                            "algorithm", "DDFA"),
                        type=str, help='Give the path of DDFA folder')
    parser.add_argument('-save_path', default=os.path.join("opendr_internal", "projects",
                                                           "data_generation",
                                                           "",
                                                           "results"),
                        type=str, help='Give the path of results folder')
    parser.add_argument('-val_yaw', default="10 20", nargs='+', type=str, help='yaw poses list between [-90,90] ')
    parser.add_argument('-val_pitch', default="30 40", nargs='+', type=str,
                        help='pitch poses list between [-90,90] ')
    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')
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
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--bbox_init', default='two', type=str, help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_2d_img', default='true', type=str2bool, help='whether to save 3d rendered image')
    parser.add_argument('--dump_param', default='true', type=str2bool, help='whether to save param')
    parser.add_argument('--dump_lmk', default='true', type=str2bool, help='whether to save landmarks')
    parser.add_argument('--save_dir', default='./algorithm/DDFA/results', type=str, help='dir to save result')
    parser.add_argument('--save_lmk_dir', default='./example', type=str, help='dir to save landmark result')
    parser.add_argument('--img_list', default='./txt_name_batch.txt', type=str, help='test image list file')
    parser.add_argument('--rank', default=0, type=int, help='used when parallel run')
    parser.add_argument('--world_size', default=1, type=int, help='used when parallel run')
    parser.add_argument('--resume_idx', default=0, type=int)
    args = parser.parse_args()
    synthetic = MultiviewDataGeneration(args)
    synthetic.eval()

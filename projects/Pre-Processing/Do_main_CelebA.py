#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as mpplot
import matplotlib.image as mpimg
import main
import inference
import argparse
import face_alignment
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool 
import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import face_alignment
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
from tqdm import tqdm
from shutil import copyfile
import csv
import itertools


list_im=[]
images = []
#STAGE No1 : detect faces and fitting to 3d mesh by main.py execution
#'''
parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')
parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
parser.add_argument('--dump_res', default='true', type=str2bool, help='whether write out the visualization image')
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
args = parser.parse_args()

rootdir = "/home/ekakalets/Downloads/Single_View/img_align_celeba/"
print("START")
a = open("file_list_celebB.txt", "w")

current_directory_path = os.path.abspath(rootdir)
print(current_directory_path)
N1=60001
N2=70001
N=5000
with open("/home/ekakalets/Documents/identity_CelebA.csv") as f:
   reader1=csv.reader(f)
   #for i in range(N1,N2):
        #row = next(f).strip()
   for row in itertools.islice(reader1, N1, N2):
        print(str(row))
        file=row[0]#row.split(",")[0]
        ID=row[1]#row.split(",")[1]
        print(str(file))
        print(str(ID))
        current_image_path= os.path.join(current_directory_path, file)
        
        print(current_image_path)
        current_image = cv2.imread(current_image_path)
        file_name =str(ID+"_"+file)
        name=os.path.join('/home/ekakalets/Documents/OPENDR/Rotate-and-Render/3ddfa/example/Images', file_name)
        cv2.imwrite(name, current_image)
        list_im.append(name)
        a.write(str(ID+"_"+file) + os.linesep)
args.files=list_im.copy()
a.close()
list_im.clear()           
main.main(args) 
#'''

#STAGE No2: Landmarks Output with inference.py execution
im_list2 = []

rootdir2 ="/home/ekakalets/Documents/OPENDR/Rotate-and-Render/3ddfa/example/Images"
d = open(os.path.join('/home/ekakalets/Documents/OPENDR/Rotate-and-Render/3ddfa/example/', 'realign_lmk'), "w")

print('START3')
parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
parser.add_argument('--bbox_init', default='two', type=str,
                        help='one|two: one-step bbox initialization or two-step')
parser.add_argument('--dump_2d_img', default='true', type=str2bool, help='whether to save 3d rendered image')
parser.add_argument('--dump_param', default='true', type=str2bool, help='whether to save param')
parser.add_argument('--dump_lmk', default='true', type=str2bool, help='whether to save landmarks')
parser.add_argument('--save_dir', default='./results', type=str, help='dir to save result')
parser.add_argument('--save_lmk_dir', default='./example', type=str, help='dir to save landmark result')
parser.add_argument('--img_list', default='./txt_name_batch.txt', type=str, help='test image list file')
parser.add_argument('--img_prefix', default=os.path.abspath(rootdir2), type=str, help='test image prefix')
parser.add_argument('--rank', default=0, type=int, help='used when parallel run')
parser.add_argument('--world_size', default=1, type=int, help='used when parallel run')
parser.add_argument('--resume_idx', default=0, type=int)

args = parser.parse_args()
current_directory_path = os.path.abspath(rootdir2)
args.img_prefix=current_directory_path
args.save_dir=os.path.abspath('/home/ekakalets/Documents/OPENDR/Rotate-and-Render/3ddfa/results')
args.save_lmk_dir=os.path.abspath('/home/ekakalets/Documents/OPENDR/Rotate-and-Render/3ddfa/example')
if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
if not os.path.exists(args.save_lmk_dir):
        os.mkdir(args.save_lmk_dir)

print(current_directory_path)
list_lfw_batch='./file_list_celebB.txt'
dst=os.path.join(args.save_lmk_dir,"file_list.txt")
copyfile(list_lfw_batch, dst)
b = open("txt_name_batch.txt", "w")
 #c = open(os.path.join(args.save_lmk_dir,"file_list.txt"), "w")
#for file in os.listdir(rootdir2 ):
#   File_split=file.split("_")[1]
#   with open(list_lfw_batch) as f:
#        img_list = [x.strip() for x in f.readlines()]
        
#        for img_idx, img_fp in enumerate(tqdm(img_list)):
#          print(File_split) 
#          if img_fp == str(File_split):
#            im_list2.append( str(file))
#            b.write(str(file) + os.linesep)
            #c.write(str(file) + os.linesep)
print(im_list2)
args.img_list= list_lfw_batch
#open('./txt_name_batch.txt', 'w').close()
#b.close()
#c.close()
args.dump_lmk = 'true'
im_list2.clear()
inference.main(args)
with open(os.path.join(args.save_lmk_dir, 'realign_lmk_')) as f:
        img_list = [x.strip() for x in f.readlines()]
        for img_idx, img_fp in enumerate(tqdm(img_list)): 
            d.write(img_fp + os.linesep)
d.close()
#'''

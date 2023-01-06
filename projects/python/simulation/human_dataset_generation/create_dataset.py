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

import os
import argparse
import gc
from data_generator import DataGenerator
import pyglet
import pickle
import json
from datetime import date
import cv2
import csv
import numpy as np


def generate_data(csv_dt_path='./csv/data2.csv', models_dir='./3D_models', back_imgs_dir='./background_imgs',
                  models_dict_path='./3D_models/models_dict.pkl', back_imgs_dict_path='./background_imgs/imgs_dict.pkl',
                  csv_tr_path=None, dataset_dir=None, placement_colors='./background_images/Cityscapes/locations_colormap.txt'):

    # Pixel colors for roads/sidewalks/terrain
    with open(placement_colors) as csvfile:
        labels_pm = []
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            row_ints = [int(i) for i in row]
            labels_pm.append(np.array(row_ints))

    data_gen = DataGenerator(models_dir, back_imgs_dir, csv_dt_path=csv_dt_path, model_dict_path=models_dict_path,
                             back_imgs_dict_path=back_imgs_dict_path, csv_tr_path=csv_tr_path,
                             data_out_dir=dataset_dir, placement_colors=labels_pm)
    pyglet.clock.schedule(data_gen.update)
    pyglet.app.run()
    pyglet.clock.unschedule(data_gen.update)
    pyglet.app.exit()
    data_gen.close()
    data = data_gen.get_data()
    gc.collect()
    return data


def generate_data_ids(data_path='./dataset', dict_path='./dataset/data_ids.pkl', splits=['train', 'test']):
    f = []
    for i in range(len(splits)):
        for (dir_path, dirnames, filenames) in os.walk(os.path.join(data_path, splits[i], 'images')):
            f.extend(filenames)
            break
    dict_ids = []
    for i in range(len(f)):
        dict_id = {
            'id': int(f[i].split('.')[0].split('_')[1]),
            'filename': f[i]
        }
        dict_ids.append(dict_id)
    with open(dict_path, 'wb') as handle:
        pickle.dump(dict_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)


def to_coco_format(annots_dir, model_ids_path, data_ids_path, split, json_path, annot_id):
    with open(data_ids_path, 'rb') as pkl_file:
        data_ids = pickle.load(pkl_file)
    # with open(model_ids_path, 'rb') as pkl_file:
    #     model_ids = pickle.load(pkl_file)
    today = date.today()
    info = {
        "description": "AUTH dataset",
        "version": "1.0",
        "contribution": "AUTH",
        "data_created": today.strftime("%d/%m/%Y")
    }
    images = []
    print(len(data_ids))
    annots = []
    for (dir_path, dirnames, filenames) in os.walk(os.path.join(annots_dir, split, 'labels')):
        annots.extend(filenames)
        break

    for i in range(len(annots)):
        img_name = os.path.splitext(annots[i])[0] + '.png'
        img = cv2.imread(os.path.join(annots_dir, split, 'images', img_name))
        for j in range(len(data_ids)):
            if data_ids[j]['filename'] == img_name:
                data_id = data_ids[j]['id']
        image = {
            "file_name": os.path.splitext(annots[i])[0] + '.png',
            "id": data_id,
            'height': img.shape[0],
            'width': img.shape[1],

        }
        images.append(image)
    categories = []
    for i in range(1):
        category = {
            "supercategory": "person",
            "name": "person",
            "id": 1,
            "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                          "left_elbow", "right_elbow",
                          "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
                          "right_ankle"],
            "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                         [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        }
        categories.append(category)
    annotations = []

    annots_csv_paths = []
    for (dir_path, dirnames, filenames) in os.walk(os.path.join(annots_dir, split, 'labels')):
        annots_csv_paths.extend(filenames)
        break
    for i in range(len(annots_csv_paths)):
        with open(os.path.join(annots_dir, split, 'labels', annots_csv_paths[i])) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            cnt = 0
            for j in range(len(data_ids)):
                if os.path.splitext(data_ids[j]['filename'])[0] == os.path.splitext(annots_csv_paths[i])[0]:
                    img_id = data_ids[j]['id']
            for row in csv_reader:
                if cnt == 0:
                    data_lab = []
                    for j in range(len(row)):
                        data_lab.append(row[j])
                elif cnt > 0:
                    model_name = row[2]
                    # model_id = row[3]
                    bb_x = int(float(row[4]))
                    bb_y = int(float(row[5]))
                    bb_width = int(float(row[6]))
                    bb_height = int(float(row[7]))
                    joints = []
                    ids = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
                    for k in ids:
                        joint = [row[8 + k * 2], row[8 + 2 * k + 1], 2]
                        joints.append(joint)
                    joints = [int(val) for sublist in joints for val in sublist]
                    annot = {
                        "id": annot_id,
                        # "category_id": int(model_id),
                        "category_id": 1,
                        "image_id": img_id,
                        "bbox": [bb_x, bb_y, bb_width, bb_height],
                        "model_name": model_name,
                        "keypoints": joints,
                        "num_keypoints": 17,
                        "iscrowd": 0,
                        "area": bb_width * bb_height
                    }
                    annotations.append(annot)
                    annot_id = annot_id + 1
                cnt = cnt + 1
    data_full = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(json_path, 'w') as jsonfile:
        json.dump(data_full, jsonfile)
    return annot_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-models_dir', type=str, default='./human_models')
    parser.add_argument('-back_imgs_out', type=str, default='./background_images/CityScapes/out')
    parser.add_argument('-models_dict_path', type=str, default='./human_models/model_ids.pkl')
    parser.add_argument('-back_imgs_dict_path', type=str, default='./background_images/CityScapes/img_ids.pkl')
    parser.add_argument('-csv_dir', type=str, default='./csv')
    parser.add_argument('-dataset_dir', type=str, default='./dataset')
    parser.add_argument('-placement_colors', type=str, default='./background_images/CityScapeslocations_colormap.txt')

    opt = parser.parse_args()

    # dataset splits
    splits = ['train', 'test']
    data_dict_path = os.path.join(opt.dataset_dir, 'data_ids.pkl')

    for j in range(len(splits)):
        csv_paths = []
        for (dir_path, dirnames, filenames) in os.walk(os.path.join(opt.csv_dir, splits[j])):
            csv_paths.extend(filenames)
            break
        for i in range(0, len(csv_paths)):
            generate_data(csv_dt_path=os.path.join(os.path.join(opt.csv_dir, splits[j]), csv_paths[i]),
                          models_dir=opt.models_dir,
                          back_imgs_dir=opt.back_imgs_out, dataset_dir=opt.dataset_dir,
                          models_dict_path=opt.models_dict_path, back_imgs_dict_path=opt.back_imgs_dict_path)

    # generate IDs for the generated data
    generate_data_ids(opt.dataset_dir, data_dict_path, splits)

    # Export annotation to COCO format
    annot_id = 0
    for j in range(len(splits)):
        json_path = os.path.join(opt.dataset_dir, splits[j], "auth_" + splits[j] + '.json')
        annot_id = to_coco_format(opt.dataset_dir, opt.models_dict_path, data_dict_path, splits[j], json_path, annot_id)

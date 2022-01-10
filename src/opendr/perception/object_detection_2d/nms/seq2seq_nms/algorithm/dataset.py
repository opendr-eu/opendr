
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


from opendr.engine.datasets import Dataset
import os
from urllib.request import urlretrieve
import time
from zipfile import ZipFile
import tarfile
import gdown
import shutil
import json
import cv2
import pickle
import numpy as np
import math
from tqdm import tqdm


class Dataset_NMS(Dataset):
    def __init__(self, path, dataset_name, split):
        super().__init__()

        available_dataset = ['COCO', 'PETS', 'CrowdHuman']
        if dataset_name not in available_dataset:
            except_str = 'Unsupported dataset: ' + dataset_name + '. Currently available are:'
            for j in range(len(available_dataset)):
                except_str = except_str + ' \'' + available_dataset[j] + '\''
                if j < len(available_dataset) - 1:
                    except_str = except_str + ','
            except_str = except_str + '.'
            raise ValueError(except_str)

        self.dataset_name = dataset_name
        self.split = split
        # self.__prepare_dataset()
        self.path = os.path.join(path, dataset_name)
        self.src_data = []
        if self.dataset_name == "PETS":
            if not os.path.exists(os.path.join(self.path, 'images/S1/L1')):
                self.download(
                    'http://ftp.cs.rdg.ac.uk/pub/PETS2009/Crowd_PETS09_dataset/a_data/Crowd_PETS09/S1_L1.tar.bz2',
                    download_path=os.path.join(self.path, 'images'), file_format="tar.bz2", create_dir=True)
            if not os.path.exists(os.path.join(self.path, 'images/S1/L2')):
                self.download(
                    'http://ftp.cs.rdg.ac.uk/pub/PETS2009/Crowd_PETS09_dataset/a_data/Crowd_PETS09/S1_L2.tar.bz2',
                    download_path=os.path.join(self.path, 'images'), file_format="tar.bz2", create_dir=True)
            if not os.path.exists(os.path.join(self.path, 'images/S2/L1')):
                self.download(
                    'http://ftp.cs.rdg.ac.uk/pub/PETS2009/Crowd_PETS09_dataset/a_data/Crowd_PETS09/S2_L1.tar.bz2',
                    download_path=os.path.join(self.path, 'images'), file_format="tar.bz2", create_dir=True)
            if not os.path.exists(os.path.join(self.path, 'images/S2/L2')):
                self.download(
                    'http://ftp.cs.rdg.ac.uk/pub/PETS2009/Crowd_PETS09_dataset/a_data/Crowd_PETS09/S2_L2.tar.bz2',
                    download_path=os.path.join(self.path, 'images'), file_format="tar.bz2", create_dir=True)
            if not os.path.exists(os.path.join(self.path, 'images/S2/L3')):
                self.download(
                    'http://ftp.cs.rdg.ac.uk/pub/PETS2009/Crowd_PETS09_dataset/a_data/Crowd_PETS09/S2_L3.tar.bz2',
                    download_path=os.path.join(self.path, 'images'), file_format="tar.bz2", create_dir=True)
            if not os.path.exists(os.path.join(self.path, 'images/S3/Multiple_Flow')):
                self.download(
                    'http://ftp.cs.rdg.ac.uk/pub/PETS2009/Crowd_PETS09_dataset/a_data/Crowd_PETS09/S3_MF.tar.bz2',
                    download_path=os.path.join(self.path, 'images'), file_format="tar.bz2", create_dir=True)
            splits = ['train', 'val', 'test']
            if self.split not in splits:
                raise ValueError(self.split + ' is not available...')
            if not os.path.exists(os.path.join(self.path, 'annotations', 'pets_' + self.split + '.json')):
                self.download('http://datasets.d2.mpi-inf.mpg.de/hosang17cvpr/PETS_annotations_json.zip',
                              download_path=os.path.join(self.path, 'annotations'), file_format="zip",
                              create_dir=True)
            if not os.path.exists(os.path.join(self.path, 'data_' + self.split + '_pets.pkl')):
                if not os.path.exists(
                        os.path.join(self.path, 'detections', 'PETS-' + self.split + '_siyudpm_dets.idl')):
                    self.download('http://datasets.d2.mpi-inf.mpg.de/hosang17cvpr/PETS_detections.zip',
                                  download_path=os.path.join(self.path, 'detections'), file_format="zip",
                                  create_dir=True)
                if not os.path.exists(os.path.join(self.path, 'annotations', 'PETS-' + self.split + '.idl')):
                    self.download('http://datasets.d2.mpi-inf.mpg.de/hosang17cvpr/PETS_annotations.zip',
                                  download_path=os.path.join(self.path, 'annotations'), file_format="zip",
                                  create_dir=True)
                with open(os.path.join(self.path, 'detections', 'PETS-' + self.split + '_siyudpm_dets.idl')) as fp_dt, \
                        open(os.path.join(self.path, 'annotations', 'PETS-' + self.split + '.idl')) as fp_gt:
                    print('Preparing PETS ' + self.split + ' set...')
                    current_id = 0
                    number_samples = 1696
                    if self.split == 'val':
                        current_id = 1696
                        number_samples = 240
                    elif self.split == 'test':
                        current_id = 1936
                        number_samples = 436
                    pbarDesc = "Overall progress"
                    pbar = tqdm(desc=pbarDesc, total=number_samples)
                    line_dt = fp_dt.readline()
                    line_gt = fp_gt.readline()
                    while line_dt and line_gt:
                        data_dt = line_dt.replace(':', ' ')
                        data_gt = line_gt.replace(':', ' ')
                        remove_strings = ['PETS09-', '\"', ':', '(', ')', ',', '', ';']
                        for j in range(len(remove_strings)):
                            data_dt = data_dt.replace(remove_strings[j], '')
                        for j in range(len(remove_strings)):
                            data_gt = data_gt.replace(remove_strings[j], '')
                        data_dt = data_dt.split()
                        data_gt = data_gt.split()
                        filename_dt = data_dt[0][0:2] + '/' + data_dt[0][2:]
                        if filename_dt[0:6] == 'S2/L1/':
                            filename_dt = filename_dt.replace('img/00', 'Time_12-34/View_001/frame_')
                            num = int(filename_dt[-8:-4]) - 1
                            filename_dt = filename_dt[:-8] + str(num).zfill(4) + '.jpg'
                        if filename_dt[0:6] == 'S2/L2/':
                            filename_dt = filename_dt.replace('img/00', 'Time_14-55/View_001/frame_')
                            num = int(filename_dt[-8:-4]) - 1
                            filename_dt = filename_dt[:-8] + str(num).zfill(4) + '.jpg'
                        if filename_dt[0:2] == 'S3':
                            filename_dt = filename_dt.replace('_MF', 'Multiple_Flow')

                        filename_gt = data_gt[0][0:2] + '/' + data_gt[0][2:]
                        if filename_gt[0:6] == 'S2/L1/':
                            filename_gt = filename_gt.replace('img/00', 'Time_12-34/View_001/frame_')
                            num = int(filename_gt[-8:-4]) - 1
                            filename_gt = filename_gt[:-8] + str(num).zfill(4) + '.jpg'
                        if filename_gt[0:6] == 'S2/L2/':
                            filename_gt = filename_gt.replace('img/00', 'Time_14-55/View_001/frame_')
                            num = int(filename_gt[-8:-4]) - 1
                            filename_gt = filename_gt[:-8] + str(num).zfill(4) + '.jpg'
                        if filename_gt[0:2] == 'S3':
                            filename_gt = filename_gt.replace('_MF', 'Multiple_Flow')
                        if filename_gt != filename_dt:
                            raise ValueError('Errors in files...')
                        img = cv2.imread(os.path.join(self.path, 'images/', filename_dt))
                        dt_boxes = []
                        for i in range(1, (len(data_dt)), 5):
                            dt_box = np.array((float(data_dt[i]), float(data_dt[i + 1]), float(data_dt[i + 2]),
                                               float(data_dt[i + 3]), 1 / (1 + math.exp(- float(data_dt[i + 4])))))
                            dt_boxes.append(dt_box)
                        gt_boxes = []
                        for i in range(1, (len(data_gt)), 5):
                            gt_box = np.array((float(data_gt[i]), float(data_gt[i + 1]), float(data_gt[i + 2]),
                                               float(data_gt[i + 3])))
                            gt_boxes.append(gt_box)
                        self.src_data.append({
                            'id': current_id,
                            'filename': filename_dt,
                            'resolution': img.shape[0:2][::-1],
                            'gt_boxes': [np.asarray([]), np.asarray(gt_boxes)],
                            'dt_boxes': [np.asarray([]), np.asarray(dt_boxes)]
                        })
                        current_id = current_id + 1
                        pbar.update(1)
                        line_dt = fp_dt.readline()
                        line_gt = fp_gt.readline()
                    pbar.close()
                    with open(os.path.join(self.path, 'data_' + self.split + '_pets.pkl'), 'wb') as handle:
                        pickle.dump(self.src_data, handle, protocol=pickle.DEFAULT_PROTOCOL)
            else:
                with open(os.path.join(self.path, 'data_' + self.split + '_pets.pkl'), 'rb') as fp_dt:
                    self.src_data = pickle.load(fp_dt)

            self.classes = ['background', 'human']
            self.class_ids = {'background': 0, 'human': 1}

        elif self.dataset_name == "CrowdHuman":
            if not os.path.exists(os.path.join(self.path, 'images/train')):
                os.makedirs(os.path.join(self.path, 'images/train'), exist_ok=True)
                urls = ['https://drive.google.com/uc?export=download&confirm=YZB1&id=134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y',
                        'https://drive.google.com/u/0/uc?id=17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla',
                        'https://drive.google.com/u/0/uc?id=1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW']
                outputs = ['CrowdHuman_train01.zip', 'CrowdHuman_train02.zip', 'CrowdHuman_train03.zip']
                for i in range(0, len(urls)):
                    gdown.download(urls[i], outputs[i], quiet=False)
                    zip_path = os.path.join('.', outputs[i])
                    with ZipFile(zip_path, 'r') as zip_ref:
                        download_path = os.path.join(self.path, 'images', 'train')
                        zip_ref.extractall(download_path)
                    os.remove(zip_path)

            if not os.path.exists(os.path.join(self.path, 'images/val')):
                os.makedirs(os.path.join(self.path, 'images/val'), exist_ok=True)
                url = 'https://drive.google.com/u/0/uc?id=18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO'
                output = 'CrowdHuman_val.zip'
                gdown.download(url, output, quiet=False)
                zip_path = os.path.join('.', output)
                with ZipFile(zip_path, 'r') as zip_ref:
                    download_path = os.path.join(self.path, 'images', 'val')
                    zip_ref.extractall(download_path)
                os.remove(zip_path)

            if not os.path.exists(os.path.join(self.path, 'data_train_crowdhuman.pkl')):
                # Download detections from FTP server

                # Download annotations from official CrowdHuman GoogleDrive repo
                if not os.path.exists(os.path.join(self.path, 'annotations', 'annotation_train.odgt')):
                    os.makedirs(os.path.join(self.path, 'annotations'), exist_ok=True)
                    urls = ['https://drive.google.com/u/0/uc?id=1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3&export=download',
                            'https://drive.google.com/u/0/uc?id=10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL&export=download']
                    outputs = ['annotation_train.odgt', 'annotation_val.odgt']
                    gdown.download(urls[0], outputs[0], quiet=False)
                    shutil.move(os.path.join('.', outputs[0]),
                                os.path.join('.', 'datasets', 'CrowdHuman', 'annotations', 'train'))
                    gdown.download(urls[1], outputs[1], quiet=False)
                    shutil.move(os.path.join('.', outputs[1]),
                                os.path.join('.', 'datasets', 'CrowdHuman', 'annotations', 'val'))

                with open(os.path.join(self.path, 'annotations', 'annotation_train.odgt')) as fp_gt, open(
                        os.path.join(self.path, 'detections', 'det_data_train_crowdhuman.pkl'), 'rb') as fp_dt:
                    line = fp_gt.readline()
                    data_dt = pickle.load(fp_dt)
                    i = 0
                    while line:
                        annotations = json.loads(line)
                        if data_dt[i]['id'] != annotations['ID']:
                            continue
                        img = cv2.imread(os.path.join(self.path, 'images/train', annotations['ID'] + '.jpg'))
                        gt_boxes = []
                        for j in range(len(annotations['gtboxes'])):
                            if annotations['gtboxes'][j]['tag'] == 'person':
                                gt_box = annotations['gtboxes'][j]['fbox']
                                gt_box[2] = gt_box[2] + gt_box[0]
                                gt_box[3] = gt_box[3] + gt_box[1]
                                gt_boxes.append(gt_box)
                        self.src_data.append({
                            'id': annotations['ID'],
                            'filename': annotations['ID'] + '.jpg',
                            'resolution': img.shape[0:2][::-1],
                            'gt_boxes': np.asarray(gt_boxes),
                            'dt_boxes': data_dt[i]['dt_boxes'],
                        })
                        line = fp_gt.readline()
                        i = i + 1
                with open(os.path.join(self.path, 'data_train_crowdhuman.pkl'), 'wb') as handle:
                    pickle.dump(self.src_data, handle, protocol=pickle.DEFAULT_PROTOCOL)
            else:
                with open(os.path.join(self.path, 'data_train_crowdhuman.pkl'), 'rb') as fp_dt:
                    self.src_data = pickle.load(fp_dt)

    @staticmethod
    def download(
            url, download_path, dataset_sub_path=".", file_format="zip", create_dir=False):

        if create_dir:
            os.makedirs(download_path, exist_ok=True)

        print("Downloading dataset from", url, "to", download_path)

        start_time = 0
        last_print = 0

        def reporthook(count, block_size, total_size):
            nonlocal start_time
            nonlocal last_print
            if count == 0:
                start_time = time.time()
                last_print = start_time
                return

            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            if time.time() - last_print >= 1:
                last_print = time.time()
                print(
                    "\r%d MB, %d KB/s, %d seconds passed" %
                    (progress_size / (1024 * 1024), speed, duration),
                    end=''
                )

        if file_format == "zip":
            zip_path = os.path.join(download_path, "dataset.zip")
            urlretrieve(url, zip_path, reporthook=reporthook)
            print()
            print("Extracting data from zip file")
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            os.remove(zip_path)
        elif file_format == "tar.bz2":
            tar_path = os.path.join(download_path, "dataset.tar.bz2")
            urlretrieve(url, tar_path, reporthook=reporthook)
            print()

            def members(tf):
                l = len("Crowd_PETS09/")
                for member in tf.getmembers():
                    if member.path.startswith("Crowd_PETS09/"):
                        member.path = member.path[l:]
                        yield member

            with tarfile.open(tar_path, "r:bz2") as tar:
                tar.extractall(path=download_path, members=members(tar))
            tar.close()
            os.remove(tar_path)
        else:
            raise ValueError("Unsupported file_format: " + file_format)

    # def __prepare_dataset(self):
    #    seq_root = os.path.join(self.path, "images/train")
    #    label_root = os.path.join(self.path, "labels/test")

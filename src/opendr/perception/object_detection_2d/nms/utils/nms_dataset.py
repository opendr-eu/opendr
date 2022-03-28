# Copyright 2020-2022 OpenDR European Project
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

from opendr.engine.datasets import Dataset
from opendr.engine.data import Image
from opendr.perception.object_detection_2d.datasets.transforms import BoundingBoxListToNumpyArray
from opendr.engine.constants import OPENDR_SERVER_URL
from pycocotools.coco import COCO
import os
from urllib.request import urlretrieve
import ssl
import time
from zipfile import ZipFile
import tarfile
import pickle
import numpy as np
import math
from tqdm import tqdm
import gc


class Dataset_NMS(Dataset):
    def __init__(self, path=None, dataset_name=None, split=None, use_ssd=True, device='cuda'):
        super().__init__()
        available_dataset = ['COCO', 'PETS', 'TEST_MODULE']
        self.dataset_sets = {'train': None,
                             'val': None,
                             'test': None}
        if dataset_name not in available_dataset:
            except_str = 'Unsupported dataset: ' + dataset_name + '. Currently available are:'
            for j in range(len(available_dataset)):
                except_str = except_str + ' \'' + available_dataset[j] + '\''
                if j < len(available_dataset) - 1:
                    except_str = except_str + ','
            except_str = except_str + '.'
            raise ValueError(except_str)

        ssl._create_default_https_context = ssl._create_unverified_context
        self.dataset_name = dataset_name
        self.split = split
        # self.__prepare_dataset()
        self.path = os.path.join(path, dataset_name)
        self.src_data = []
        if self.dataset_name == "PETS":
            self.detector = 'JPD'
            self.detector_type = 'default'
            if use_ssd:
                self.detector = 'SSD'
                self.detector_type = 'custom'

            self.dataset_sets['train'] = 'train'
            self.dataset_sets['val'] = 'val'
            self.dataset_sets['test'] = 'test'
            if self.dataset_sets[self.split] is None:
                raise ValueError(self.split + ' split is not available...')

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
            if not os.path.exists(
                    os.path.join(self.path, 'annotations', 'pets_' + self.dataset_sets[self.split] + '.json')):
                self.download('http://datasets.d2.mpi-inf.mpg.de/hosang17cvpr/PETS_annotations_json.zip',
                              download_path=os.path.join(self.path, 'annotations'), file_format="zip",
                              create_dir=True)
            pkl_filename = os.path.join(self.path,
                                        'data_' + self.detector + '_' + self.dataset_sets[self.split] + '_pets.pkl')
            if not os.path.exists(pkl_filename):
                ssd = None
                if use_ssd:
                    from opendr.perception.object_detection_2d.ssd.ssd_learner import SingleShotDetectorLearner
                    ssd = SingleShotDetectorLearner(device=device)
                    ssd.download(".", mode="pretrained")
                    ssd.load("./ssd_default_person", verbose=True)
                if not os.path.exists(
                        os.path.join(self.path, 'detections',
                                     'PETS-' + self.dataset_sets[self.split] + '_siyudpm_dets.idl')):
                    self.download('http://datasets.d2.mpi-inf.mpg.de/hosang17cvpr/PETS_detections.zip',
                                  download_path=os.path.join(self.path, 'detections'), file_format="zip",
                                  create_dir=True)
                if not os.path.exists(
                        os.path.join(self.path, 'annotations', 'PETS-' + self.dataset_sets[self.split] + '.idl')):
                    self.download('http://datasets.d2.mpi-inf.mpg.de/hosang17cvpr/PETS_annotations.zip',
                                  download_path=os.path.join(self.path, 'annotations'), file_format="zip",
                                  create_dir=True)
                with open(os.path.join(self.path, 'annotations',
                                       'PETS-' + self.dataset_sets[self.split] + '.idl')) as fp_gt:
                    fp_dt = None
                    if self.detector_type == 'default':
                        fp_dt = open(os.path.join(self.path, 'detections',
                                                  'PETS-' + self.dataset_sets[self.split] + '_siyudpm_dets.idl'))
                    print('Preparing PETS ' + self.dataset_sets[self.split] + ' set...')
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
                    if self.detector_type == 'default':
                        line_dt = fp_dt.readline()
                    line_gt = fp_gt.readline()
                    while line_gt:
                        remove_strings = ['PETS09-', '\"', ':', '(', ')', ',', '', ';']
                        data_gt = line_gt.replace(':', ' ')
                        for j in range(len(remove_strings)):
                            data_gt = data_gt.replace(remove_strings[j], '')
                        data_gt = data_gt.split()
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

                        if self.detector_type == 'default':
                            data_dt = line_dt.replace(':', ' ')
                            for j in range(len(remove_strings)):
                                data_dt = data_dt.replace(remove_strings[j], '')
                            data_dt = data_dt.split()
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
                            if filename_gt != filename_dt:
                                raise ValueError('Errors in files...')

                        img = Image.open(os.path.join(self.path, 'images/', filename_gt))

                        dt_boxes = []
                        if self.detector_type == 'default':
                            for i in range(1, (len(data_dt)), 5):
                                dt_box = np.array((float(data_dt[i]), float(data_dt[i + 1]), float(data_dt[i + 2]),
                                                   float(data_dt[i + 3]), 1 / (1 + math.exp(- float(data_dt[i + 4])))))
                                dt_boxes.append(dt_box)
                        else:
                            bboxes_list = ssd.infer(img, threshold=0.0, custom_nms=None, nms_thresh=0.975,
                                                    nms_topk=6000, post_nms=6000)
                            bboxes_list = BoundingBoxListToNumpyArray()(bboxes_list)
                            bboxes_list = bboxes_list[bboxes_list[:, 4] > 0.015]
                            bboxes_list = bboxes_list[np.argsort(bboxes_list[:, 4]), :][::-1]
                            bboxes_list = bboxes_list[:5000, :]
                            for b in range(len(bboxes_list)):
                                dt_boxes.append(np.array([bboxes_list[b, 0], bboxes_list[b, 1], bboxes_list[b, 2],
                                                          bboxes_list[b, 3], bboxes_list[b, 4][0]]))
                        gt_boxes = []
                        for i in range(1, (len(data_gt)), 5):
                            gt_box = np.array((float(data_gt[i]), float(data_gt[i + 1]), float(data_gt[i + 2]),
                                               float(data_gt[i + 3])))
                            gt_boxes.append(gt_box)
                        self.src_data.append({
                            'id': current_id,
                            'filename': os.path.join('images', filename_gt),
                            'resolution': img.opencv().shape[0:2][::-1],
                            'gt_boxes': [np.asarray([]), np.asarray(gt_boxes)],
                            'dt_boxes': [np.asarray([]), np.asarray(dt_boxes)]
                        })
                        current_id = current_id + 1
                        pbar.update(1)
                        if self.detector_type == 'default':
                            line_dt = fp_dt.readline()
                        line_gt = fp_gt.readline()
                    pbar.close()
                    if self.detector_type == 'default':
                        fp_dt.close()
                    elif self.detector == 'SSD':
                        del ssd
                        gc.collect()
                    with open(pkl_filename, 'wb') as handle:
                        pickle.dump(self.src_data, handle, protocol=pickle.DEFAULT_PROTOCOL)
            else:
                with open(pkl_filename, 'rb') as fp_pkl:
                    self.src_data = pickle.load(fp_pkl)

            self.classes = ['background', 'human']
            self.class_ids = [-1, 1]
            self.annotation_file = 'pets_' + self.dataset_sets[self.split] + '.json'
        elif self.dataset_name == "COCO":
            self.dataset_sets['train'] = 'train'
            self.dataset_sets['val'] = 'minival'
            self.dataset_sets['test'] = 'valminusminival'
            if self.dataset_sets[self.split] is None:
                raise ValueError(self.split + ' split is not available...')
            elif self.dataset_sets[self.split] == 'train':
                imgs_split = 'train2014'
            else:
                imgs_split = 'val2014'
            self.detector = 'FRCN'
            self.detector_type = 'default'
            ssd = None
            if use_ssd:
                self.detector = 'SSD'
                self.detector_type = 'custom'
                from opendr.perception.object_detection_2d.ssd.ssd_learner import SingleShotDetectorLearner
                ssd = SingleShotDetectorLearner(device=device)
                ssd.download(".", mode="pretrained")
                ssd.load("./ssd_default_person", verbose=True)
            if not os.path.exists(os.path.join(self.path, imgs_split)):
                self.download('http://images.cocodataset.org/zips/' + imgs_split + '.zip',
                              download_path=os.path.join(self.path), file_format="zip",
                              create_dir=True)
            pkl_filename = os.path.join(self.path, 'data_' + self.detector + '_' +
                                        self.dataset_sets[self.split] + '_coco.pkl')
            if not os.path.exists(pkl_filename):
                if not os.path.exists(os.path.join(self.path, 'annotations', 'instances_' +
                                                                             self.dataset_sets[self.split] +
                                                                             '2014.json')):
                    if self.dataset_sets[self.split] == 'train':
                        ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
                        self.download(ann_url, download_path=os.path.join(self.path), file_format="zip",
                                      create_dir=True)
                    else:
                        if self.dataset_sets[self.split] == 'minival':
                            ann_url = 'https://dl.dropboxusercontent.com/s/o43o90bna78omob/' \
                                      'instances_minival2014.json.zip?dl=0'
                        else:
                            ann_url = 'https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/' \
                                      'instances_valminusminival2014.json.zip?dl=0'
                        self.download(ann_url, download_path=os.path.join(self.path, 'annotations'), file_format="zip",
                                      create_dir=True)
                if not os.path.exists(os.path.join(self.path, 'detections', 'coco_2014_' +
                                                                            self.dataset_sets[self.split] +
                                                                            '_FRCN_train.pkl')):
                    self.download('http://datasets.d2.mpi-inf.mpg.de/hosang17cvpr/coco_2014_FRCN.tar.gz',
                                  download_path=os.path.join(self.path, 'detections'), file_format='tar.gz',
                                  create_dir=True)
                with open(os.path.join(self.path, 'detections',
                                       'coco_2014_' + self.dataset_sets[self.split] + '_FRCN_train.pkl'), 'rb') as f:
                    dets_default = pickle.load(f, encoding='latin1')
                annots = COCO(annotation_file=os.path.join(self.path, 'annotations', 'instances_' +
                                                           self.dataset_sets[self.split] + '2014.json'))
                pbarDesc = "Overall progress"
                pbar = tqdm(desc=pbarDesc, total=len(dets_default[1]))
                for i in range(len(dets_default[1])):
                    dt_boxes = []
                    img_info = annots.loadImgs([dets_default[1][i]])[0]
                    img = Image.open(os.path.join(self.path, imgs_split, img_info["file_name"]))
                    if self.detector_type == 'default':
                        dt_boxes = dets_default[0][1][i]
                    elif self.detector == 'SSD':
                        bboxes_list = ssd.infer(img, threshold=0.0, custom_nms=None, nms_thresh=0.975,
                                                nms_topk=6000, post_nms=6000)
                        bboxes_list = BoundingBoxListToNumpyArray()(bboxes_list)
                        if bboxes_list.shape[0] > 0:
                            bboxes_list = bboxes_list[bboxes_list[:, 4] > 0.015]
                        if bboxes_list.shape[0] > 0:
                            bboxes_list = bboxes_list[np.argsort(bboxes_list[:, 4]), :][::-1]
                            bboxes_list = bboxes_list[:5000, :]
                        for b in range(len(bboxes_list)):
                            dt_boxes.append(np.array([bboxes_list[b, 0], bboxes_list[b, 1], bboxes_list[b, 2],
                                                      bboxes_list[b, 3], bboxes_list[b, 4][0]]))
                    dt_boxes = np.asarray(dt_boxes)
                    annots_in_frame = annots.loadAnns(
                        annots.getAnnIds(imgIds=[dets_default[1][i]], catIds=[1], iscrowd=False))
                    gt_boxes = []
                    for j in range(len(annots_in_frame)):
                        gt_boxes.append(annots_in_frame[j]['bbox'])
                    gt_boxes = np.asarray(np.asarray(gt_boxes))
                    if gt_boxes.shape[0] > 0:
                        gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
                        gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]
                    self.src_data.append({
                        'id': dets_default[1][i],
                        'filename': os.path.join(imgs_split, img_info["file_name"]),
                        'resolution': [img_info['width'], img_info['height']],
                        'gt_boxes': [np.asarray([]), gt_boxes],
                        'dt_boxes': [np.asarray([]), dt_boxes]
                    })
                    pbar.update(1)
                pbar.close()
                if self.detector == 'SSD':
                    del ssd
                    gc.collect()
                with open(pkl_filename, 'wb') as handle:
                    pickle.dump(self.src_data, handle, protocol=pickle.DEFAULT_PROTOCOL)
            else:
                with open(pkl_filename, 'rb') as fp_pkl:
                    self.src_data = pickle.load(fp_pkl)
            self.classes = ['background', 'person']
            self.class_ids = [-1, 1]
            self.annotation_file = 'instances_' + self.dataset_sets[self.split] + '2014.json'
        elif self.dataset_name == "TEST_MODULE":
            self.dataset_sets['train'] = 'test'
            self.dataset_sets['val'] = 'test'
            self.dataset_sets['test'] = 'test'
            if self.dataset_sets[self.split] is None:
                raise ValueError(self.split + ' split is not available...')
            pkl_filename = os.path.join(self.path, 'test_module.pkl')
            if not os.path.exists(pkl_filename):
                data_url = OPENDR_SERVER_URL + '/perception/object_detection_2d/nms/datasets/test_module.zip'
                self.download(data_url, download_path=os.path.join(self.path).replace("TEST_MODULE", ""), file_format="zip",
                              create_dir=True)
            with open(pkl_filename, 'rb') as fp_pkl:
                self.src_data = pickle.load(fp_pkl)
            self.classes = ['background', 'person']
            self.class_ids = [-1, 1]
            self.annotation_file = 'test_module_anns.json'

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
        elif file_format == "tar.bz2" or file_format == "tar.gz":
            tar_path = os.path.join(download_path, "dataset." + file_format)
            urlretrieve(url, tar_path, reporthook=reporthook)
            print()

            def members(tf):
                l = len("Crowd_PETS09/")
                for member in tf.getmembers():
                    if member.path.startswith("Crowd_PETS09/"):
                        member.path = member.path[l:]
                        yield member

            with tarfile.open(tar_path, "r:" + file_format.split('.')[1]) as tar:
                if file_format == "tar.bz2":
                    tar.extractall(path=download_path, members=members(tar))
                else:
                    tar.extractall(path=download_path)
            tar.close()
            os.remove(tar_path)
        else:
            raise ValueError("Unsupported file_format: " + file_format)

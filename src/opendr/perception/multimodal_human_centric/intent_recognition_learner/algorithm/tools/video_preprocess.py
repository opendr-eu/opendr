'''
This code is based on https://github.com/thuiar/MIntRec under MIT license:

MIT License

Copyright (c) 2022 Hanlei Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
'''


import os
import torch
import numpy as np
import json
import pickle
import argparse

from torch import nn
from tqdm import tqdm

from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_checkpoint_path', type=str,
                        default='mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
                        help="The directory of the detection checkpoint path.")
    parser.add_argument('--detection_config_path', type=str,
                        default='mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
                        help="The directory of the detection configuration path.")
    parser.add_argument('--video_data_path', type=str,
                        default='MIA/datasets/video_data',
                        help="The directory of the video data path.")
    parser.add_argument('--video_feats_path', type=str,
                        default='video_feats_test.pkl',
                        help="The directory of the video features path.")
    parser.add_argument('--frames_path', type=str,
                        default='MIA/datasets/human_annotations/screenshots',
                        help="The directory of human-annotated frames with bbox.")
    parser.add_argument('--speaker_annotation_path', type=str,
                        default='MIA/datasets/human_annotations/speaker_annotations.json',
                        help="The original file of annotated speaker ids.")
    parser.add_argument('--TalkNet_speaker_path', type=str,
                        default='MIA/datasets/speaker_annotation/Talknet',
                        help="The output directory of TalkNet model.")
    parser.add_argument("--use_TalkNet", action="store_true",
                        help="whether using the annotations from TalkNet to get video features.")
    parser.add_argument("--roi_feat_size", type=int, default=7, help="The size of Faster R-CNN region of interest.")

    args = parser.parse_args()

    return args


class VideoFeature:

    def __init__(self, args):

        self.model, self.device = self._init_detection_model(args)
        self.avg_pool = nn.AvgPool2d(args.roi_feat_size)

    def _get_feats(self, args):

        if args.use_TalkNet:
            self.bbox_feats = self._get_TalkNet_features(args)

        else:
            self.bbox_feats = self._get_Annotated_features(args)

    def _save_feats(self, args):

        video_feats_path = os.path.join(args.video_data_path, args.video_feats_path)

        with open(video_feats_path, 'wb') as f:
            pickle.dump(self.bbox_feats, f)

    def _init_detection_model(self, args):

        model = init_detector(args.detection_config_path, args.detection_checkpoint_path, device='cuda:0')
        device = next(model.parameters()).device  # model device
        return model, device

    def _get_TalkNet_features(self, args):

        '''
        Input:
            args.TalkNet_speaker_path

        Output:
        The format of video features
        {
            'video_clip_id_a':[frame_a_feat, frame_b_feat, ..., frame_N_feat],
            'video_clip_id_b':[xxx]
        }
        '''

        video_feats = {}
        error_cnt = 0
        error_path = 0

        for video_clip_name in tqdm(os.listdir(args.TalkNet_speaker_path), desc='Video'):
            frames_path = os.path.join(args.TalkNet_speaker_path, video_clip_name, 'pyframes')
            bestperson_path = os.path.join(args.TalkNet_speaker_path, video_clip_name, 'pywork', 'best_persons.npy')

            if not os.path.exists(bestperson_path):
                error_path += 1
                continue

            bestpersons = np.load(bestperson_path)

            for frame, bbox in tqdm(enumerate(bestpersons), desc='Frame'):

                if (bbox[0] == 0) and (bbox[1] == 0) and (bbox[2] == 0) and (bbox[3] == 0):
                    error_cnt += 1
                    continue

                frame_name = str('%06d' % frame)
                frame_path = os.path.join(frames_path, frame_name + '.jpg')

                """
                img = cv2.imread(img_ath)
                height, width, channel = img.shape
                roi = [0, 0, width, height]
                """

                roi = bbox.tolist()
                roi.insert(0, 0.)

                bbox_feat = self._extract_roi_feats(self.model, self.device, frame_path, roi)
                bbox_feat = self._average_pooling(bbox_feat)
                bbox_feat = bbox_feat.detach().cpu().numpy()

                if video_clip_name not in video_feats.keys():
                    video_feats[video_clip_name] = [bbox_feat]

                else:
                    video_feats[video_clip_name].append(bbox_feat)

        print('The number of error annotations is {}'.format(error_cnt))
        print('The number of error paths is {}'.format(error_path))

        return video_feats

    def _get_Annotated_features(self, args):

        '''
        Input:
            args.video_data_path
            args.speaker_annotation_path
            args.frames_path

        Output:
        The format of video features
        {
            'video_clip_id_a':[frame_a_feat, frame_b_feat, ..., frame_N_feat],
            'video_clip_id_b':[xxx]
        }
        '''

        speaker_annotation_path = os.path.join(args.video_data_path, args.speaker_annotation_path)
        speaker_annotations = json.load(open(speaker_annotation_path, 'r'))

        video_feats = {}
        error_cnt = 0

        try:
            for key in tqdm(speaker_annotations.keys(), desc='Frame'):

                if 'bbox' not in speaker_annotations[key].keys():
                    error_cnt += 1
                    continue

                roi = speaker_annotations[key]['bbox'][:4]
                roi.insert(0, 0.)

                frame_name = '_'.join(key.strip('.jpg').split('_')[:-1])
                frame_path = os.path.join(args.frames_path, frame_name + '.jpg')

                bbox_feat = self._extract_roi_feats(self.model, self.device, frame_path, roi)
                bbox_feat = self._average_pooling(bbox_feat)
                bbox_feat = bbox_feat.detach().cpu().numpy()

                video_clip_name = '_'.join(key.strip('.jpg').split('_')[:-2])

                if video_clip_name not in video_feats.keys():
                    video_feats[video_clip_name] = [bbox_feat]
                else:
                    video_feats[video_clip_name].append(bbox_feat)

        except Exception as e:
            print(e)

        print('The number of error annotations is {}'.format(error_cnt))

        return video_feats

    def _extract_roi_feats(self, model, device, file_path, roi):
        roi = torch.tensor([roi]).to(device)
        cfg = model.cfg
        data = dict(img_info=dict(filename=file_path), img_prefix=None)
        test_pipeline = Compose(cfg.data.test.pipeline)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [device])[0]

        img = data['img'][0]
        x = model.extract_feat(img)

        bbox_feat = model.roi_head.bbox_roi_extractor(
            x[:model.roi_head.bbox_roi_extractor.num_inputs], roi)

        return bbox_feat

    def _average_pooling(self, x):
        """
        Args:
        x: dtype: numpy.ndarray
        """
        x = self.avg_pool(x)
        x = x.flatten(1)
        return x


if __name__ == '__main__':

    args = parse_arguments()

    args.use_TalkNet = True
    video_data = VideoFeature(args)
    video_data._get_feats(args)
    video_data._save_feats(args)

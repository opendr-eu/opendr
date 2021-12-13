"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""

import argparse
import os
import numpy as np
from opendr.perception.skeleton_based_action_recognition.algorithm.datasets.kinetics_feeder import KineticsFeeder
import pickle
from tqdm import tqdm
import pandas
from pathlib import Path


KINETICS400_CLASSES = pandas.read_csv(Path(__file__).parent /
                                      'kinetics400_classes.csv', verbose=True, index_col=0).to_dict()["name"]


def gendata(data_path, label_path,
            data_out_path, label_out_path,
            num_person_in=5,  # observe the first 5 persons
            num_person_out=2,  # then choose 2 persons with the highest score
            max_frame=300):
    feeder = KineticsFeeder(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    fp = np.zeros((len(sample_name), 3, 300, 18, num_person_out), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data, label = feeder[i]
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    np.save(data_out_path, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='./data/kinetics_raw')
    parser.add_argument(
        '--out_folder', default='./data/kinetics')
    arg = parser.parse_args()

    part = ['val', 'train']
    for p in part:
        print('kinetics ', p)
        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        data_path = '{}/kinetics_{}'.format(arg.data_path, p)
        label_path = '{}/kinetics_{}_label.json'.format(arg.data_path, p)
        data_out_path = '{}/{}_data_joint.npy'.format(arg.out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

        gendata(data_path, label_path, data_out_path, label_out_path)

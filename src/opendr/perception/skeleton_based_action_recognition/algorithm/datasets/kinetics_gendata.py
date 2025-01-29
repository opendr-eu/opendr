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

from numpy.lib.format import open_memmap
from opendr.perception.skeleton_based_action_recognition.algorithm.datasets.kinetics_feeder import KineticsFeeder

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
    
    
def gendata_mmap(data_path, label_path,
            data_out_path, label_out_path,
            num_person_in=5,  # observe the first 5 persons
            num_person_out=2,  # then choose 2 persons with the highest score
            max_frame=300,
            chunk_size=128):

    feeder = KineticsFeeder(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame
    )

    sample_name = feeder.sample_name
    num_samples = len(sample_name)
    sample_label = [None] * num_samples  # avoid appending

    fp_shape = (num_samples, 3, max_frame, 18, num_person_out)  # configure open_memmap
    fp_dtype = np.float32

    # create empty file in disk
    fp = open_memmap(
        data_out_path,
        mode='w+',
        dtype=fp_dtype,
        shape=fp_shape
    )
    
    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        current_size = end_idx - start_idx

        for i in tqdm(range(current_size), desc=f"Chunk {start_idx}-{end_idx}", leave=False):
        
            idx_global = start_idx + i
            data, label = feeder[idx_global]
            T = data.shape[1]

            fp[idx_global, :, :T, :, :] = data
            sample_label[idx_global] = label
            
        fp.flush()  # write to disk
    del fp

    
    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, sample_label), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.'
    )
    parser.add_argument('--data_path', default='./data/kinetics_raw')
    parser.add_argument('--out_folder', default='./data/kinetics')
    parser.add_argument('--use_mmap', action='store_true',
                        help="Whether to use memory-mapped numpy arrays.")
    parser.add_argument('--chunk_size', type=int, default=128,
                        help="Number of samples processed in each chunk.")
    arg = parser.parse_args()

    part = ['val', 'train']
    for p in part:
        print('Kinetics', p)
        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        data_path = f'{arg.data_path}/kinetics_{p}'
        label_path = f'{arg.data_path}/kinetics_{p}_label.json'
        data_out_path = f'{arg.out_folder}/{p}_data_joint.npy'
        label_out_path = f'{arg.out_folder}/{p}_label.pkl'

        if not arg.use_mmap:
            gendata(
                data_path, label_path,
                data_out_path, label_out_path,
                num_person_in=5,
                num_person_out=2,
                max_frame=300
            )
        else:
            gendata_mmap(
                data_path, label_path,
                data_out_path, label_out_path,
                num_person_in=5,
                num_person_out=2,
                max_frame=300,
                chunk_size=arg.chunk_size
            )
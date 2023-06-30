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


from moviepy.editor import VideoFileClip
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import os
import pickle
import argparse
import librosa
import torch

__all__ = ['AudioFeature']


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_video_path', type=str, default='raw_video', help="The directory of the raw video path.")
    parser.add_argument('--audio_data_path', type=str, default='audio_data', help="The directory of the audio data path.")
    parser.add_argument('--raw_audio_path', type=str, default='raw_audio', help="The directory of the raw audio path.")
    parser.add_argument("--audio_feats_path", type=str, default='audio_feats.pkl', help="The directory of audio features.")

    args = parser.parse_args()

    return args


class AudioFeature:

    def __init__(self, args):

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        self.__get_raw_audio(args)

        audio_feats = self.__gen_feats_from_audio(args, use_wav2vec2=True)
        self.__save_audio_feats(args, audio_feats)

    def __get_raw_audio(self, args):

        raw_audio_path = os.path.join(args.audio_data_path, args.raw_audio_path)

        if not os.path.exists(raw_audio_path):
            os.makedirs(raw_audio_path)

        for season in tqdm(os.listdir(args.raw_video_path), desc='Season'):

            episode_path = os.path.join(args.raw_video_path, season)

            for episode in tqdm(os.listdir(episode_path), desc='Episode'):

                clip_path = os.path.join(episode_path, episode)
                audio_data_path = os.path.join(raw_audio_path, season, episode)
                if not os.path.exists(audio_data_path):
                    os.makedirs(audio_data_path)

                for clip in tqdm(os.listdir(clip_path), desc='Clip'):

                    video_path = os.path.join(clip_path, clip)
                    print(video_path)

                    video_name = clip.split('.')[0]
                    video_segments = VideoFileClip(video_path)
                    audio = video_segments.audio
                    audio.write_audiofile(os.path.join(audio_data_path, video_name + ".wav"))

    def __gen_feats_from_audio(self, args, use_wav2vec2=False):

        audio_feats = {}
        raw_audio_path = os.path.join(args.audio_data_path, args.raw_audio_path)

        for s_path in tqdm(os.listdir(raw_audio_path), desc='Season'):

            s_path_dir = os.path.join(raw_audio_path, s_path)

            for e_path in tqdm(os.listdir(s_path_dir), desc='Episode'):
                e_path_dir = os.path.join(s_path_dir, e_path)

                for file in tqdm(os.listdir(e_path_dir), desc='Clip'):

                    audio_id = '_'.join([s_path, e_path, file[:-4]])
                    read_file_path = os.path.join(e_path_dir, file)

                    if use_wav2vec2:
                        wav2vec2_feats = self.__process_audio(read_file_path)
                        audio_feats[audio_id] = wav2vec2_feats
                    else:
                        mfcc = self.__process_audio(read_file_path)
                        audio_feats[audio_id] = mfcc

        return audio_feats

    def __process_audio(self, read_file_path):

        y, sr = librosa.load(read_file_path, sr=16000)
        audio_feats = self.processor(y, sampling_rate=sr, return_tensors="pt").input_values
        with torch.no_grad():
            audio_feats = self.model(audio_feats).last_hidden_state.squeeze(0)

        return audio_feats

    def __save_audio_feats(self, args, audio_feats):

        audio_feats_path = os.path.join(args.audio_data_path, args.audio_feats_path)

        with open(audio_feats_path, 'wb') as f:
            pickle.dump(audio_feats, f)


if __name__ == '__main__':

    args = parse_arguments()
    audio_data = AudioFeature(args)

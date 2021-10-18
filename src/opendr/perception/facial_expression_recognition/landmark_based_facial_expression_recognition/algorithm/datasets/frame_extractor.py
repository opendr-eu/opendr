# Copyright 2020-2021 OpenDR European Project
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
import subprocess
import cv2
import argparse


def getFrame(vidcap, sec, framespth, count):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        cv2.imwrite(os.path.join(framespth, "image" + str(count) + ".jpg"), image)
    return hasFrames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='video frame extractor')
    parser.add_argument("-i", "--video_folder", required=True, default='./data/AFEW_videos/',
                        description="path to input video")
    parser.add_argument("-i", "--frames_folder", required=True, default='./data/AFEW/',
                        description="path to input video")
    arg = vars(parser.parse_args())
    classes = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Disgust', 'Neutral']
    part = ['Train', 'Val']
    for p in part:
        for c in classes:
            video_path = arg.video_folder + '/{}/{}'.format(p, c)
            frames_path = arg.frames_folder + '/{}/{}'.format(p, c)
            for root, _, files in os.walk(video_path):
                count_files = 0
                for file in files:
                    if '.avi' in file:
                        count_files += 1
                        vidpth_avi = os.path.join(root, file)
                        vidpth_mp4 = os.path.join(video_path, 'mp4vids')
                        framespth = os.path.join(frames_path, str(count_files))
                        if not os.path.exists(vidpth_mp4):
                            os.makedirs(vidpth_mp4)
                        command = "ffmpeg -i {input} {output}".format(input=vidpth_avi,
                                                                      output=vidpth_mp4 + str(count_files) + ".mp4")
                        subprocess.call(command, shell=True)  # convert .avi to .mp4
                        vidcap = cv2.VideoCapture(vidpth_mp4 + str(count_files) + ".mp4")
                        # extract frames from each video
                        if not os.path.exists(framespth):
                            os.makedirs(framespth)
                        sec = 0
                        frameRate = 0.5  # it captures frames every 0.5 second
                        count = 0
                        success = getFrame(vidcap, sec, framespth, count)
                        while success:
                            count += 1
                            sec = sec + frameRate
                            sec = round(sec, 2)
                            success = getFrame(vidcap, sec, framespth, count)

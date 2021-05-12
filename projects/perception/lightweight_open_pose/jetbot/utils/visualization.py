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

from os.path import exists
from opendr.perception.pose_estimation.lightweight_open_pose.utilities import draw
import cv2
import threading
import numpy as np
import time
import sys
from flask import Flask, render_template, Response
from os.path import join


def fall_handler_fn(images, output_file=None, output_path="results"):
    """
    Function that can be used for handling a fall. The specific function just displays the last 4 snapshots and
    then goes into an endless loop to stop the controller.
    @param images: images to display when a fall has been detected
    @param output_file: location to save the output image
    @param output_path: folder to save the resulting images
    """
    images = [np.uint8(x) for x in images]
    h, w, c = images[0].shape
    image = np.zeros((h, w * 2, c), dtype=np.uint8)
    image[:h, :w, :] = images[0]
    image[:h, w:, :] = images[1]

    if output_file:
        print("Writing output files...")
        cv2.imwrite(join(output_path, output_file + '_1.png'), images[0])
        cv2.imwrite(join(output_path, output_file + '_1.png'), images[1])

    print("Fall detected! Exiting...")
    sys.exit(0)


class Visualizer:

    def __init__(self, opendr_logo='static/opendr.png', stream=True, local_display=False, video_writer=None):
        """
        Class that provides functionality regarding displaying various statistics during the detections
        @param opendr_logo: path to the opendr logo
        @param stream: if set to true, then streams the annotated video to localhost:5000
        @param local_display: if set to true, then displays an opencv window with the stream
        @param video_writer: video writer to use
        """
        self.local_display = local_display
        if exists(opendr_logo):
            self.opendr_logo = cv2.resize(cv2.imread(opendr_logo), (240, 100))
        else:
            self.fopendr_logo = None
        self.video_writer = video_writer
        self.start_time = time.time()

        self.frame = None
        # Use flask to stream the video feed
        if stream:
            app = Flask('video_steamer')

            @app.route('/video_feed')
            def video_feed():
                return Response(self.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

            @app.route('/')
            def index():
                return render_template('index.html')
            # Start flask server
            try:
                threading.Thread(target=lambda: app.run(debug=False, host="0.0.0.0")).start()
            except Exception as e:
                print("Flask error occurred =  ", e)

    def gen_frames(self):  # generate frame by frame from camera

        while True:
            if self.frame is None:
                time.sleep(0.01)
                continue
            else:
                frame = cv2.resize(np.float32(self.frame), (960, 540))
                ret, buffer = cv2.imencode('.jpg', np.uint8(frame))
                frame_raw = buffer.tobytes()
                time.sleep(0.01)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_raw + b'\r\n')

    def visualization_handler(self, img, pose, statistics):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale, fontColor, lineType = 1, (255, 255, 255), 1

        # Draw the pose
        img = np.uint8(img)
        img_down = cv2.resize(img, (800, 600))

        cv2.putText(img, 'state: %s' % (statistics['state']), (10, 20), font, fontScale, fontColor, lineType)
        if 'control_left' in statistics:
            cv2.putText(img, 'commands [left: %.2f right : %.2f]' % (
                statistics['control_left'], statistics['control_right']),
                        (10, 20 + 30), font, fontScale, fontColor, lineType)
            if 'fall_confidence' in statistics:
                cv2.putText(img, 'pose size: %.2f, offset: %.2f, fall_confidence: %.2f' % (
                    statistics['size'], statistics['offset'],
                    statistics['fall_confidence']),
                            (10, 20 + 2 * 30), font, fontScale, fontColor, lineType)
            else:
                if statistics['size'] is not None and statistics['offset'] is not None:
                    cv2.putText(img, 'pose size: %.2f, offset: %.2f,' % (statistics['size'], statistics['offset'],),
                                (10, 20 + 2 * 30), font, fontScale, fontColor, lineType)
            if 'fall' in statistics and statistics['fall']:
                cv2.putText(img, 'FALL DETECTED!', (10, 20 + 4 * 30), font, fontScale, fontColor, lineType)
            elif 'control' in statistics:
                cv2.putText(img, 'performing control... ', (10, 20 + 4 * 30), font, fontScale, fontColor, lineType)
            elif ('skipped' in statistics and statistics['skipped']) or pose is None:
                cv2.putText(img, 'low confidence!', (10, 20 + 4 * 30), font, fontScale, fontColor, lineType)

        cv2.putText(img, 'wall time: %.2f s' % (time.time() - self.start_time), (10, 20 + 3 * 30), font, fontScale,
                    fontColor, lineType)

        # Add logo on the lower corner
        logo = self.opendr_logo
        if logo is not None:
            img[-logo.shape[0]:, -logo.shape[1]:, :] = logo

        if pose:
            draw(img_down, pose)
            img_down = cv2.resize(img_down, (396, 216))
        else:
            img_down = cv2.resize(img_down, (396, 216))

        cv2.putText(img, 'Detected Pose', (70, img.shape[0] - 230), font, fontScale, fontColor, lineType)

        img[-216:, :396, :] = img_down

        if 'heatmap' in statistics and statistics['heatmap'] is not None:
            heatmap = statistics['heatmap']
            cv2.putText(img, 'Heatmap', (430, img.shape[0] - 180), font, fontScale, fontColor, lineType)
            img[-heatmap.shape[0]:, 400:400 + heatmap.shape[1], 0] = statistics['heatmap']
            img[-heatmap.shape[0]:, 400:400 + heatmap.shape[1], 1] = statistics['heatmap']
            img[-heatmap.shape[0]:, 400:400 + heatmap.shape[1], 2] = statistics['heatmap']

        if self.local_display:
            cv2.imshow('', img)
            cv2.waitKey(1)
        if self.video_writer is not None:
            self.video_writer.write(img)
            # Write a few additional frames when fall is detected
            if 'fall' in statistics and statistics['fall']:
                for i in range(20):
                    self.video_writer.write(img)

        # If flask is used, then save the last frame
        self.frame = img

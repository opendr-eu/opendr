# Copyright 2020-2024 OpenDR European Project
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
import cv2
import os
import torch

from opendr.perception.object_detection_2d import RetinaFaceLearner
from opendr.perception.object_detection_2d.datasets.transforms import\
    BoundingBoxListToNumpyArray
from opendr.perception.face_recognition import FaceRecognitionLearner

facedetector = RetinaFaceLearner(backbone='mnet', device='cuda')
facedetector.download(".", mode="pretrained")
facedetector.load("./retinaface_mnet")

recognizer = FaceRecognitionLearner(device='cuda', backbone='mobilefacenet', mode='backbone_only')
recognizer.download(path=".")
recognizer.load(".")
recognizer.fit_reference('./cropped_images_path', save_path="./save_path", create_new=True)

new_data_path = recognizer.database_path
cam = cv2.VideoCapture(0)
cv2.namedWindow("face recognition")
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2
avg_counter = 0
img_counter = 0
features_to_keep = {}
features_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    bounding_boxes = facedetector.infer(frame)
    if bounding_boxes:
        bounding_boxes_ = BoundingBoxListToNumpyArray()(bounding_boxes)
        boxes = bounding_boxes_[:, :4]
        for idx, box in enumerate(boxes):
            (startX, startY, endX, endY) = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            img = frame[startY:endY, startX:endX]
            result = recognizer.infer(img)
            avg_counter += 1
            if result.description != 'Not found':
                if result.confidence < 0.40:
                    color = (0, 125, 0)
                    if avg_counter % 50 == 0:
                        img_counter += 1
                        features_counter += 1
                        path = os.path.join(new_data_path, 'New', str(result.description))
                        if not os.path.exists(path):
                            os.makedirs(path)
                        cv2.imwrite(os.path.join(path, str(img_counter) + '.jpg'), img)
                        features, closest_id, distance = recognizer.feature_extraction(img)
                        if result.description not in features_to_keep:
                            features_to_keep[result.description] = [features]
                        else:
                            features_to_keep[result.description].append(features)
                else:
                    color = (0, 255, 0)
                    features_sum = torch.zeros(1, recognizer.embedding_size).to(recognizer.device)
                    if result.description in features_to_keep:
                        for item in features_to_keep[result.description]:
                            features_sum += item
                        recognizer.database[result.description] = \
                            (features_sum / len(features_to_keep[result.description]))

            else:
                color = (0, 0, 255)
            img = cv2.rectangle(frame, (startX, startY), (endX, endY), color, thickness)
            img = cv2.putText(img, result.description, (startX, endY - 10), font,
                              fontScale, color, thickness, cv2.LINE_AA)
    else:
        img = frame
    cv2.imshow("face recognition", img)
    cv2.waitKey(1)

cam.release()

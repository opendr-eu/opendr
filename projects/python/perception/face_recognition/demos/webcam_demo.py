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


import cv2

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

cam = cv2.VideoCapture(0)
cv2.namedWindow("face recognition")
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2
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
            if result.description != 'Not found':
                color = (0, 255, 0)
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

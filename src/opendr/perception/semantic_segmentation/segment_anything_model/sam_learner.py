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

import time
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

from segment_anything import sam_model_registry, SamPredictor

from opendr.engine.learners import Learner
from opendr.engine.target import BoundingBox, BoundingBoxList


class SamLearner(Learner):

    def __init__(self, backbone="", device="cuda"):
        super(SamLearner, self).__init__(device=device)

        sam_checkpoint = "sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        url = f"https://dl.fbaipublicfiles.com/segment_anything/{sam_checkpoint}"
        file_path = f"./{sam_checkpoint}"
        if not os.path.exists(file_path):
            print("Downloading model...")
            urlretrieve(url, file_path)
            print("Download complete")
        else:
            print("Pretrained model already downloaded.")

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.model = SamPredictor(sam)

    def fit(self, dataset, val_dataset=None, logging_path='', silent=True, verbose=True):
        pass

    def eval(self, dataset):
        pass

    def infer(self, img, bounding_box_prompt):
        # TODO check for type, model expects cv2 RGB, wrap with opendr image
        # image = cv2.imread('./images/custom_test_img.jpg')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.model.set_image(img)  # Around 1 fps @1050ti
        # print(self.model.features.shape)  # Image embeddings

        # Prompt with points
        # input_point = np.array([[93, 91], [383, 99], [208, 297], [30, 134]])
        # input_label = np.array([1, 1, 1, 1])
        #
        # masks, scores, logits = self.model.predict(
        #     point_coords=input_point,
        #     point_labels=input_label,
        #     multimask_output=False,
        # )
        # plt.figure(figsize=(10, 10))
        # plt.imshow(img)
        # self.show_mask(masks, plt.gca())
        # self.show_points(input_point, input_label, plt.gca())
        # plt.axis('off')
        # plt.show()
        multimask= False
        # Prompt with box
        x0, y0 = int(bounding_box_prompt.left), int(bounding_box_prompt.top)
        x1, y1 = int(bounding_box_prompt.left + bounding_box_prompt.width), \
            int(bounding_box_prompt.top + bounding_box_prompt.height)
        bbox_prompt = np.array([x0, y0, x1, y1])

        # # Additional point prompt from the center of the bbox
        # input_point = np.array([[bounding_box_prompt.left + (bounding_box_prompt.width // 2),
        #                          bounding_box_prompt.top + (bounding_box_prompt.height // 2)]])
        # input_label = np.array([1])
        #
        # cv2.circle(img, (int(input_point[0][0]), int(input_point[0][1])), 3, [255, 0, 255], -1)
        start_time = time.perf_counter()
        masks, scores, logits = self.model.predict(  # ~7fps @1050ti
            point_coords=None,
            point_labels=None,
            box=bbox_prompt[None, :],
            multimask_output=multimask,
        )

        end_time = time.perf_counter()
        fps = 1.0 / (end_time - start_time)
        print("Predict fps:", fps)
        if multimask:
            masks = masks[np.argmax(scores)]  # Choose the model's best mask

        return masks, scores, logits, bbox_prompt

    def save(self, path):
        pass

    def load(self, path):
        pass

    def optimize(self, target_device):
        pass

    def reset(self):
        pass

    def get_mask(self, mask, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            color = np.array([30, 144, 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape((1, 1, -1))
        # ax.imshow(mask_image)
        return np.array(mask_image)

    def get_points(self, coords, labels, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        return pos_points, neg_points
        # ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
        #            linewidth=1.25)
        # ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
        #            linewidth=1.25)

    def get_box(self, box):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        # ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        return x0, y0, w, h

    def draw(self, image, box, mask):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), [255, 255, 255], thickness=2)
        alpha = 0.5
        beta = 1.0 - alpha
        mask = self.get_mask(mask)
        image = np.asarray(image, np.uint8)
        mask = np.asarray(mask, np.uint8)
        image_blended = cv2.addWeighted(image, alpha, mask, beta, 0.0)
        return image_blended
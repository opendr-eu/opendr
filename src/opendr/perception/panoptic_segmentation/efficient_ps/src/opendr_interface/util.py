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

from typing import List
import numpy as np
from mmdet.core.evaluation import cityscapes_originalIds


def collate_fn(batch):
    return batch

def prepare_results_for_evaluation(results: List[np.ndarray]):
    originalIds = cityscapes_originalIds()

    processed_results = []
    for result in results:
        images = []
        annotations = []
        pan_pred, cat_pred, meta = result
        pan_pred, cat_pred = pan_pred.numpy(), cat_pred.numpy()
        # imgName = meta[0]['filename'].split('/')[-1]
        # imageId = imgName.replace(".png", "")
        # inputFileName = imgName
        # outputFileName = imgName.replace(".png", "_panoptic.png")
        # images.append({"id": imageId,
        #                "width": int(pan_pred.shape[1]),
        #                "height": int(pan_pred.shape[0]),
        #                "file_name": inputFileName})

        pan_format = np.zeros((pan_pred.shape[0], pan_pred.shape[1], 3), dtype=np.uint8)

        panPredIds = np.unique(pan_pred)
        segmInfo = []
        for panPredId in panPredIds:
            if cat_pred[panPredId] == 255:
                continue
            elif cat_pred[panPredId] <= 10:
                semanticId = segmentId = originalIds[cat_pred[panPredId]]
            else:
                semanticId = originalIds[cat_pred[panPredId]]
                segmentId = semanticId * 1000 + panPredId

            is_crowd = 0
            categoryId = semanticId

            mask = pan_pred == panPredId
            color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
            pan_format[mask] = color

            area = np.sum(mask)

            # bbox computation for a segment
            hor = np.sum(mask, axis=0)
            hor_idx = np.nonzero(hor)[0]
            x = hor_idx[0]
            width = hor_idx[-1] - x + 1
            vert = np.sum(mask, axis=1)
            vert_idx = np.nonzero(vert)[0]
            y = vert_idx[0]
            height = vert_idx[-1] - y + 1
            bbox = [int(x), int(y), int(width), int(height)]

            segmInfo.append({"id": int(segmentId),
                             "category_id": int(categoryId),
                             "area": int(area),
                             "bbox": bbox,
                             "iscrowd": is_crowd})
        processed_results.append(segmInfo)
        # annotations.append({'image_id': imageId,
        #                     'file_name': outputFileName,
        #                     "segments_info": segmInfo})
        #
        # Image.fromarray(pan_format).save(os.path.join(base_path, outputFileName))
        # d = {'images': images,
        #      'annotations': annotations,
        #      'categories': {}}

    return processed_results

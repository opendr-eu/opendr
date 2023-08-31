FSeq2-NMS
======

This folder contains an implementation of the FSeq2-NMS algorithm, for neural Non-Maximum Suppression in visual person detection. If one uses any part of this implementation in their work, they are kindly asked to cite the following paper/s:

- C. Symeonidis, I. Mademlis, I. Pitas and N. Nikolaidis, "[Efficient Feature Extraction for Non-Maximum Suppression in Visual Person Detection](https://ieeexplore.ieee.org/document/10095074)" in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10095074.
- C. Symeonidis, I. Mademlis, I. Pitas and N. Nikolaidis, "[Neural Attention-Driven Non-Maximum Suppression for Person Detection](https://ieeexplore.ieee.org/abstract/document/10107719)" in IEEE Transactions on Image Processing, vol. 32, pp. 2454-2467, 2023, doi: 10.1109/TIP.2023.3268561.

TABLE-1: Parameters of the performed benchmarking.  
|||
|:-----------:|:----------------------:|
|  **Dataset**   |   PETS  |
|  **Detector**  |   SSD   |
| **Detector's training dataset** | PETS |
| **Pre-processing IoU Threshold** | 0.8 |
| **Maximum number of outputted RoIs** | 800 |

TABLE-2: Average Precision (AP) achieved by pretrained models on the person detection task on the validation and testing sets.
| **Method**  |  **model_name / nms_threshold**  | **AP<sub>0.5</sub> on validation set** | **AP<sub>0.5</sub><sup>0.95</sup> on validation set** |**AP<sub>0.5</sub> on testing set** | **AP<sub>0.5</sub><sup>0.95</sup> on testing set** |
|:-----------:|:--------------------------------:|:--------------------------------------:|:-----------------------------------------------------:|:----------------------------------:|:--------------------------------------------------:|
|         Fast-NMS           |     nms_thres: 0.70     |             81.9%            |          34.9%         |             87.4%            |          37.0%         |
|    Soft-NMS<sub>L</sub>    |     nms_thres: 0.55     |             85.5%            |          37.1%         |             90.4%            |          39.2%         |
|    Soft-NMS<sub>G</sub>    |     nms_thres: 0.90     |             84.2%            |          37.3%         |             90.0%            |          39.6%         |
|        Cluster-NMS         |     nms_thres: 0.60     |             84.6%            |          36.0%         |             90.2%            |          38.2%         |
|  Cluster-NMS<sub>S</sub>   |     nms_thres: 0.35     |             85.1%            |          37.1%         |             90.3%            |          39.0%         |
|  Cluster-NMS<sub>D</sub>   |     nms_thres: 0.55     |             84.8%            |          35.7%         |             90.5%            |          38.1%         |
| Cluster-NMS<sub>S+D</sub>  |     nms_thres: 0.45     |             86.0%            |          37.2%         |             90.9%            |          39.2%         |
| Cluster-NMS<sub>S+D+W</sub>|     nms_thres: 0.45     |             86.0%            |          37.2%         |             90.9%            |          39.2%         |
|        Seq2Seq-NMS         | name: seq2seq_pets_ssd_pets  |             87.8%            |          38.4%         |             91.2%            |          39.5%         |
|        Fseq2-NMS           |  name: fseq2_pets_ssd_pets   |             87.8%            |          38.6%         |             91.5%            |          39.4%         |

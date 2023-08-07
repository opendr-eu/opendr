FSeq2-NMS
======

This folder contains an implementation of the FSeq2-NMS algorithm, for neural Non-Maximum Suppression in visual person detection. If one uses any part of this implementation in his/her work, he/she is kindly asked to cite the following paper/s:

- C. Symeonidis, I. Mademlis, I. Pitas and N. Nikolaidis, "[Efficient Feature Extraction for Non-Maximum Suppression in Visual Person Detection](https://ieeexplore.ieee.org/document/10095074)" in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10095074.
- C. Symeonidis, I. Mademlis, I. Pitas and N. Nikolaidis, "[Neural Attention-Driven Non-Maximum Suppression for Person Detection](https://ieeexplore.ieee.org/abstract/document/10107719)" in IEEE Transactions on Image Processing, vol. 32, pp. 2454-2467, 2023, doi: 10.1109/TIP.2023.3268561.


TABLE-1: Average Precision (AP) achieved by pretrained models on the person detection task on the validation and test sets. The maximum number or RoIs, employed for the performance evaluation was set to 800.
| **Method**  |  **Pretrained Model**  | **Dataset** | **Detector** | **Detector's training dataset** | **Parameters** | **Pre-processing IoU Threshold** | **AP<sub>0.5</sub> on validation set** | **AP<sub>0.5</sub><sup>0.95</sup> on validation set** |**AP<sub>0.5</sub> on testing set** | **AP<sub>0.5</sub><sup>0.95</sup> on testing set** |
|:-----------:|:----------------------:|:-----------:|:------------:|:-------------------------------:|:--------------:|:--------------------------------:|:------------------------------:|:--------------------------------------------:|:-----------------------------:|:---------------------------------------------:|
|  Soft-NMS<sub>L</sub>  |   seq2seq_pets_ssd   |     PETS    |      SSD     |              PETS               |     nms_thres: 0.55     |           0.8           |             85.5%            |          37.1%         |             90.4%            |          39.2%         |
|  Soft-NMS<sub>G</sub>  |   seq2seq_pets_ssd   |     PETS    |      SSD     |              PETS               |     nms_thres: 0.90     |           0.8           |             84.2%            |          37.3%         |             90.0%            |          39.6%         |
|      Seq2Seq-NMS       |   seq2seq_pets_ssd   |     PETS    |      SSD     |              PETS               |          ----           |           0.8           |             87.9%            |          38.6%         |             91.3%            |          39.7%         |
|      Fseq2-NMS         |    fseq2_pets_ssd    |     PETS    |      SSD     |              PETS               |          ----           |           0.8           |             88.2%            |          38.7%         |             91.3%            |          39.5%         |

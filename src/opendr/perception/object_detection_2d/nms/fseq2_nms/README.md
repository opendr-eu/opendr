FSeq2-NMS
======

This folder contains an implementation of the FSeq2-NMS algorithm, for neural Non-Maximum Suppression in visual person detection. If one uses any part of this implementation in his/her work, he/she is kindly asked to cite the following paper/s:

- C. Symeonidis, I. Mademlis, I. Pitas and N. Nikolaidis, "[Efficient Feature Extraction for Non-Maximum Suppression in Visual Person Detection](https://ieeexplore.ieee.org/document/10095074)" in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1-5.
- C. Symeonidis, I. Mademlis, I. Pitas and N. Nikolaidis, "[Neural Attention-Driven Non-Maximum Suppression for Person Detection](https://ieeexplore.ieee.org/abstract/document/10107719)" in IEEE Transactions on Image Processing, vol. 32, pp. 2454-2467, 2023, doi: 10.1109/TIP.2023.3268561.


TABLE-1: Average Precision (AP) achieved by pretrained models on the person detection task on the validation sets. The maximum number or RoIs, employed for the performance evaluation was set to 800.
|  **Pretrained Model**  | **Dataset** | **Detector** | **Pre-processing IoU Threshold** | **AP@0.5 on validation set** | **AP@0.5 on test set** |
|:----------------------:|:-----------:|:------------:|:--------------------------------:|:----------------------------:|:----------------------:|
|  fseq2_pets            |     PETS    |      SSD     |                0.8               |             XX.X%            |          XX.X%         |
|  fseq2_crowdhuman      |  CROWDHUMAN |      SSD     |                0.8               |               -              |          XX.X%         |

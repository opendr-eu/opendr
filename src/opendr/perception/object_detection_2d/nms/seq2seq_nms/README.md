Seq2Seq-NMS
======

This folder contains an implementation of the Seq2Seq-NMS algorithm, for neural Non-Maximum Suppression in visual person detection. If one uses any part of this implementation in his/her work, he/she is kindly asked to cite the following paper:

- C. Symeonidis, I. Mademlis, I. Pitas and N. Nikolaidis, "[Neural Attention-Driven Non-Maximum Suppression for Person Detection](https://ieeexplore.ieee.org/abstract/document/10107719)" in IEEE Transactions on Image Processing, vol. 32, pp. 2454-2467, 2023, doi: 10.1109/TIP.2023.3268561.

TABLE-1: Average Precision (AP) achieved by pretrained models on the person detection task on the validation sets. The maximum number or RoIs, employed for the performance evaluation was set to 800.
|  **Pretrained Model**  | **Dataset** | **Detector** | **Type of Appearance-based Features** | **Pre-processing IoU Threshold** | **AP@0.5 on validation set** | **AP@0.5 on test set** |
|:----------------------:|:-----------:|:------------:|:-------------------------------------:|:--------------------------------:|:----------------------------:|:----------------------:|
|  seq2seq_pets_jpd_fmod |     PETS    |      JPD     |                  FMoD                 |                0.8               |             80.2%            |          84.3%         |
|  seq2seq_pets_ssd_fmod |     PETS    |      SSD     |                  FMoD                 |                0.8               |             77.4%            |          79.1%         |
| seq2seq_coco_frcn_fmod |     COCO    |     FRCN     |                  FMoD                 |                 -                |             68.1% \*            |          67.5% \*\*         |
| seq2seq_coco_ssd_fmod  |     COCO    |      SSD     |                  FMoD                 |                 -                |             41.8% \*                 |            42.4% **      |

\* The minival set was used as validation set.<br>
\*\* The minitest set was used as test set.


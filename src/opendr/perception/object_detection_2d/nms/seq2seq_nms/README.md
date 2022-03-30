Seq2Seq-NMS
======

This folder contains an implementation of Seq2Seq-NMS [[1]](#seq2seq_nms-1).

TABLE-1: Average Precision (AP) achieved by pretrained models on the person detection task on the validation sets. The maximum number or RoIs, employed for the performance evaluation was set to 800.
|  **Pretrained Model**  | **Dataset** | **Detector** | **Type of Appearance-based Features** | **Pre-processing IoU Threshold** | **AP@0.5 on validation set** | **AP@0.5 on test set** |
|:----------------------:|:-----------:|:------------:|:-------------------------------------:|:--------------------------------:|:----------------------------:|:----------------------:|
|  seq2seq_pets_jpd_fmod |     PETS    |      JPD     |                  FMoD                 |                0.8               |             80.2%            |          84.3%         |
|  seq2seq_pets_ssd_fmod |     PETS    |      SSD     |                  FMoD                 |                0.8               |             77.4%            |          79.1%         |
| seq2seq_coco_frcn_fmod |     COCO    |     FRCN     |                  FMoD                 |                 -                |             68.1% \*            |          67.5% \*\*         |
| seq2seq_coco_ssd_fmod  |     COCO    |      SSD     |                  FMoD                 |                 -                |             41.8% \*                 |            42.4% **      |

\* The minival set was used as validation set.<br>
\*\* The minitest set was used as test set.

<a name="seq2seq_nms-1" href="https://www.techrxiv.org/articles/preprint/Neural_Attention-driven_Non-Maximum_Suppression_for_Person_Detection/16940275">[1]</a> Neural Attention-driven Non-Maximum Suppression for Person Detection, [TechRxiv](https://www.techrxiv.org/articles/preprint/Neural_Attention-driven_Non-Maximum_Suppression_for_Person_Detection/16940275).

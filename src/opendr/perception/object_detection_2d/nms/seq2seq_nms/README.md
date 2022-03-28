Seq2Seq-NMS
======

This folder contains an implementation of Seq2Seq-NMS [[1]](#seq2seq_nms-1).

TABLE-1: Average Precision (AP) achieved by pretrained models on the person detection task. The maximum number or RoIs, employed for the performance evaluation was set to 800.
|  **Pretrained Model**  | **Dataset** | **Detector** | **Type of Appearance-based Features** | **Pre-processing IoU Threshold** | **AP_0.5** |   |   |   |
|:----------------------:|:-----------:|:------------:|:-------------------------------------:|:--------------------------------:|:----------:|:-:|:-:|---|
|  seq2seq_pets_jpd_fmod |     PETS    |      JPD     |                  FMoD                 |                0.8               |            |   |   |   |
|  seq2seq_pets_ssd_fmod |     PETS    |      SSD     |                  FMoD                 |                0.8               |            |   |   |   |
| seq2seq_coco_frcn_fmod |     COCO    |     FRCN     |                  FMoD                 |                0.8               |            |   |   |   |
| seq2seq_coco_frcn_fmod |     COCO    |      SSD     |                  FMoD                 |                0.8               |            |   |   |   |

![Alt text](stats_pretrained.png?raw=true "Title")




<a name="seq2seq_nms-1" href="https://www.techrxiv.org/articles/preprint/Neural_Attention-driven_Non-Maximum_Suppression_for_Person_Detection/16940275">[1]</a> Neural Attention-driven Non-Maximum Suppression for Person Detection,
[ArXiv](https://www.techrxiv.org/articles/preprint/Neural_Attention-driven_Non-Maximum_Suppression_for_Person_Detection/16940275).

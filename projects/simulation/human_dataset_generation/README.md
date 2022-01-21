# Human-Data-Generation-Framework

This folder contains the code for generating the data described in "Efficient Realistic Data Generation Framework leveraging Deep Learning-based Human Digitization"

## Download and reformat the Cityscapes dataset

1. Download the 3D human models

```
chmod +x download_models.sh
./download_models.sh
```

2. Download the Cityscapes dataset from www.cityscapes-dataset.net <br />
    * RGB images: (a) leftImg8bit_trainvaltest.zip,  (b) leftImg8bit_trainextra.zip <br />
    * Annotation images: gtCoarse.zip <br />

The folder hierarchy should look like this:
```
├─ background_images
|  ├─ Cityscapes
|     └─ in
|     |   ├─ leftImg8Bit
|     |   └─ gtCoarse
|     └─ out
|      
...
```
3. Run the following script to reformat the Cityscapes dataset
```
python3 reformat_cityscapes.py -data_dir ./background_images/Cityscapes/in
python3 create_background_images.py -rgb_in ./background_images/Cityscapes/in/all/rgb -segm_in ./background_images/Cityscapes/in/all/segm -imgs_dir_out ./background_images/Cityscapes/out -human_colors ./background_images/Cityscapes/human_colormap.txt -placement_colors ./background_images/Cityscapes/locations_colormap.txt
``` 
4. Run the following script to generate the dataset
```
python3 create_dataset.py -models_dir ./human_models -back_imgs_out ./background_images/Cityscapes/out -csv_dir ./csv -dataset_dir ./dataset
```   
## Citation
If you make use of the dataset, please cite the following reference in any publications:
```
@inproceedings{symeonidis2021data,
  title={Efficient Realistic Data Generation Framework leveraging Deep Learning-based Human Digitization},
  author={Symeonidis, C. and Nousi, P. and Tosidis, P. and Tsampazis, K. and Passalis, N. and Tefas, A. and Nikolaidis, N.}
  booktitle={Proceedings of the International Conference on Engineering Applications of Neural Networks (EANN)},
  year={2021}
}
```

# Human-Data-Generation-Framework

This folder contains the code for generating the data described in "Efficient Realistic Data Generation Framework leveraging Deep Learning-based Human Digitization"

## Download and reformat the CityScapes dataset

1. Download the 3D human models

```
chmod +x download_models.sh
./download_models.sh
```

2. Download the CityScapes dataset from www.cityscapes-dataset.net <br />
    * RGB images: (a) leftImg8bit_trainvaltest.zip,  (b) leftImg8bit_trainextra.zip <br />
    * Annotation images: gtCoarse.zip <br />

The folder hierarchy should look like this:
```
├─ background_images
|  ├─ CityScapes
|     └─ in
|     |   ├─ leftImg8Bit
|     |   └─ gtCoarse
|     └─ out
|      
...
```
3. Run the following script to reformat the CityScapes dataset
```
python reformat_cityscapes.py -data_dir ./background_images/CityScapes/in
python create_background_images.py -rgb_in ./background_images/CityScapes/in/all/rgb -segm_in ./background_images/CityScapes/in/all/segm -imgs_dir_out ./background_images/CityScapes/out -human_colors ./background_images/CityScapes/human_colormap.txt -placement_colors ./background_images/CityScapes/locations_colormap.txt
``` 
4. Run the following script to generate the dataset
```
python create_dataset.py -models_dir ./human_models -back_imgs_out ./background_images/CityScapes/out -csv_dir ./csv -dataset_dir ./dataset
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


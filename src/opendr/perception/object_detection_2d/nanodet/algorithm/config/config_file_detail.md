# NanoDet Config File Analysis

NanoDet using [yacs](https://github.com/rbgirshick/yacs) to read YAML config file.

## Saving path

```yaml
save_dir: PATH_TO_SAVE
```

Change `save_dir` to where you want to save logs and models. If path doesn't exist, NanoDet will create it.

## Model

```yaml
model:
    arch:
        name: OneStageDetector
        backbone: xxx
        fpn: xxx
        head: xxx
```

Most detection model architecture can be divided into 3 parts: backbone, task head and connector between them (e.g., FPN, PAN).

### Backbone

```yaml
backbone:
    name: ShuffleNetV2
    model_size: 1.0x
    out_stages: [2,3,4]
    activation: LeakyReLU
    with_last_conv: False
```

NanoDet using ShuffleNetV2 as backbone. You can modify model size, output feature levels and activation function. Moreover, NanoDet provides other lightweight backbones like **GhostNet** and **MobileNetV2**. You can also add your backbone network by importing it in `nanodet/model/backbone/__init__.py`.

### FPN

```yaml
fpn:
    name: PAN
    in_channels: [116, 232, 464]
    out_channels: 96
    start_level: 0
    num_outs: 3
```

NanoDet using modified [PAN](http://arxiv.org/abs/1803.01534) (replace downsample convs with interpolation to reduce amount of computations).

`in_channels`: a list of feature map channels extracted from backbone.

`out_channels`: output feature map channel.

### Head

```yaml
head:
    name: NanoDetHead
    num_classes: 80
    input_channel: 96
    feat_channels: 96
    stacked_convs: 2
    share_cls_reg: True
    octave_base_scale: 8
    strides: [8, 16, 32]
    reg_max: 7
    norm_cfg:
      type: BN
    loss:
```

`name`: task head class name

`num_classes`: number of classes

`input_channel`: input feature map channel

`feat_channels`: channel of task head convs

`stacked_convs`: how many conv blocks use in one task head

`strides`: down sample stride of each feature map level

`reg_max`: max value of per-level l-r-t-b distance

`norm_cfg`: normalization layer setting

`loss`: adjust loss functions and weights

`assigner_cfg`: config dictionary of the assigner.

`share_cls_reg`: use same conv blocks for classification and box regression. Used in GFLHead and NanoDetHead.

`octave_base_scale`: base box scale. Used in GFLHead and NanoDetHead.

`use_depthwise`: whether to use PointWise-DepthWise or Base convolutions modules. Used in NanoDetHead and NanoDetPlusHead

`kernel_size`: size of the convolving kernel. Used in NanoDetPlusHead

`activation`: type of activation function. Used in NanoDetHead and NanoDetPlusHead

`legacy_post_process`: whether to use legacy post-processing or not.
If set to False, a faster implementation of post-processing will be used with respect to dynamic input.
Most applications will run the same with either post-processing implementations. Used in NanoDetPlusHead.

## Weight averaging

Nanodet supports weight averaging method like EMA:

```yaml
model:
  weight_averager:
    name: ExpMovingAverager
    decay: 0.9998
  arch:
    ...
```

## Data

```yaml
data:
    train:
        input_size: [320,320]
        keep_ratio: True
        cache_images: _
        multi_scale: [0.6, 1.4]
        pipeline:
    val:
        ...
```

In `data` you need to set your train and validate dataset.

`input_size`: [width, height]

`keep_ratio`: whether to maintain the original image ratio when resizing to input size.

`cache_images`: whether to cache images or not during training. "disk" option will cache images as numpy files in disk, "ram" option will cache dataset into ram.

`multi_scale`: scaling range for multi-scale training. Set to None to turn off.

`pipeline`: data preprocessing and augmentation pipeline.

## Device

```yaml
device:
    gpu_ids: [0]
    workers_per_gpu: 12
    batchsize_per_gpu: 160
    effective_batchsize: 1
```

`gpu_ids`: CUDA device id.

`workers_per_gpu`: how many dataloader processes for each gpu

`batchsize_per_gpu`: amount of images in one batch for each gpu, if -1 autobatch will determine the batchsize to be used.

`effective_batchsize`: determines the effective batch size by accumulating losses, 1 will use only batchsize_per_gpu.
## schedule

```yaml
schedule:
  resume: 0
  optimizer:
    name: SGD
    lr: 0.14
    momentum: 0.9
    weight_decay: 0.0001
  warmup:
    name: linear
    steps: 300
    ratio: 0.1
  total_epochs: 70
  lr_schedule:
    name: MultiStepLR
    milestones: [40,55,60,65]
    gamma: 0.1
  val_intervals: 10
```

Set training schedule.

`resume`: to restore # checkpoint, if 0 model start from random initialization

`load_model`: path to trained weight

`optimizer`: support all optimizer provided by pytorch.

You should adjust the `lr` with `batch_size`. Following linear scaling rule in paper *[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h5.pdf)*

`warmup`: warm up your network before training. Support `constant`, `exp` and `linear` three types of warm up.

`total_epochs`: total epochs to train

`lr_schedule`: please refer to [pytorch lr_scheduler documentation](https://pytorch.org/docs/stable/optim.html?highlight=lr_scheduler#torch.optim.lr_scheduler)

`val_intervals`: epoch interval of evaluating during training

## Evaluate

```yaml
evaluator:
  name: CocoDetectionEvaluator
  save_key: mAP
```

Currently only support COCO eval.

`save_key`: metric of best model. Support mAP, AP50, AP75....

****

`class_names`: used in visualization

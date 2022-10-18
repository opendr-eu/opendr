import copy

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head.gfl_head import GFLHead
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head.nanodet_head import NanoDetHead
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head.nanodet_plus_head import NanoDetPlusHead
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.model.head.simple_conv_head import SimpleConvHead


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")
    if name == "GFLHead":
        return GFLHead(**head_cfg)
    elif name == "NanoDetHead":
        return NanoDetHead(**head_cfg)
    elif name == "NanoDetPlusHead":
        return NanoDetPlusHead(**head_cfg)
    elif name == "SimpleConvHead":
        return SimpleConvHead(**head_cfg)
    else:
        raise NotImplementedError

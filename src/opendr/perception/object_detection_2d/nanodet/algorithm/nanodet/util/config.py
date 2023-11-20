from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util.yacs import CfgNode

cfg = CfgNode(new_allowed=True)
cfg.save_dir = "./"
# common params for NETWORK
cfg.model = CfgNode(new_allowed=True)
cfg.model.arch = CfgNode(new_allowed=True)
cfg.model.arch.ch_l = False
cfg.model.arch.backbone = CfgNode(new_allowed=True)
cfg.model.arch.fpn = CfgNode(new_allowed=True)
cfg.model.arch.head = CfgNode(new_allowed=True)

# DATASET related params
cfg.data = CfgNode(new_allowed=True)
cfg.data.train = CfgNode(new_allowed=True)
cfg.data.train.cache_images = "_"
cfg.data.val = CfgNode(new_allowed=True)
cfg.data.val.cache_images = "_"
cfg.device = CfgNode(new_allowed=True)
cfg.device.precision = 32
cfg.device.batchsize_per_gpu = -1
cfg.device.effective_batchsize = 1
# train
cfg.schedule = CfgNode(new_allowed=True)

# logger
cfg.log = CfgNode()
cfg.log.interval = 50

# testing
cfg.test = CfgNode()
# size of images for each device


def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print(cfg, file=f)

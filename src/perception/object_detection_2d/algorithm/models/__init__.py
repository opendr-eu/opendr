# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build, build_c, build_pp

def build_postprocessors(args):
    return build_pp(args)

def build_criterion(args):
    return build_c(args)

def build_model(args):
    return build(args)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Modifications Copyright 2021 - present, OpenDR European Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .mm_detr import build, build_c, build_pp


def build_postprocessors(args):
    return build_pp(args)


def build_criterion(args):
    return build_c(args)


def build_model(args, fusion_method, backbone_name):
    return build(args, fusion_method, backbone_name)

#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import OrderedDict, namedtuple
import torch
import numpy as np

import tensorrt as trt

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx as pycuda_autinit  # noqa
except ModuleNotFoundError:
    import pycuda.autoinit as pycuda_autinit  # noqa
var = pycuda_autinit

try:
    # Sometimes python does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def GiB(val):
    return val * 1 << 30


class trt_model():
    def __init__(self, engine, device="cuda"):
        self.device = device
        self.engine = engine
        self.context = engine.create_execution_context()

        self.bindings = OrderedDict()
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.output_names = []
        self.fp16 = False
        self.dynamic = False
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            if self.engine.binding_is_input(i):
                if -1 in tuple(self.engine.get_binding_shape(i)):  # dynamic
                    self.dynamic = True
                    self.context.set_binding_shape(i, tuple(self.engine.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    self.fp16 = True
                if dtype == np.int8:
                    self.int8 = True
            else:  # output
                self.output_names.append(name)
            shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.batch_size = self.bindings['data'].shape[0]  # if dynamic, this is instead max batch size

    def __call__(self, input):
        # input = input.to(memory_format=torch.contiguous_format)  # maybe slows down (check)
        if self.dynamic and input.shape != self.bindings['data'].shape:
            i = self.engine.get_binding_index('data')
            self.context.set_binding_shape(i, input.shape)  # reshape if dynamic
            self.bindings['data'] = self.bindings['data']._replace(shape=input.shape)
            for name in self.output_names:
                i = self.engine.get_binding_index(name)
                self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
        s = self.bindings['data'].shape
        assert input.shape == s, f"input size {input.shape} not equal to max model size {s}"
        self.binding_addrs['data'] = int(input.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

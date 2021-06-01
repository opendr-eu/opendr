from torch import Tensor
from typing import Callable, Optional, List

# Defined in tools/autograd/templates/python_nn_functions.cpp

fractional_max_pool2d: Callable
fractional_max_pool3d: Callable
max_pool1d: Callable
max_pool2d: Callable
max_pool3d: Callable
adaptive_max_pool1d: Callable
adaptive_max_pool2d: Callable
adaptive_max_pool3d: Callable
avg_pool2d: Callable
avg_pool3d: Callable
hardtanh_: Callable
elu_: Callable
leaky_relu_: Callable
logsigmoid: Callable
softplus: Callable
softshrink: Callable
one_hot: Callable
hardtanh: Callable
leaky_relu: Callable
hardsigmoid: Callable

# Defined in aten/src/ATen/native/mkldnn/Linear.cpp
def mkldnn_linear(input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor: ...

# Defined at aten/src/ATen/native/mkldnn/MKLDNNConversions.cpp
def mkldnn_reorder_conv2d_weight(self: Tensor, padding: List, stride: List, dilatation: List, groups: int) -> Tensor: ...
def mkldnn_reorder_conv3d_weight(self: Tensor, padding: List, stride: List, dilatation: List, groups: int) -> Tensor: ...
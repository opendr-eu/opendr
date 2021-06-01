__version__ = '0.8.2'
git_version = '2f40a483d73018ae6e1488a484c5927f2b309969'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()

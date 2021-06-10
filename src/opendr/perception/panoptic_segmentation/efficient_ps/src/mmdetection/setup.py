#!/usr/bin/env python
import os
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def find_sources(root_dir):
    sources = []
    for file in os.listdir(root_dir):
        _, ext = os.path.splitext(file)
        if ext in [".cpp", ".cu"]:
            sources.append(os.path.join(root_dir, file))

    return sources


def make_extension(name, package):
    return CUDAExtension(
        name="{}.{}._backend".format(package, name),
        sources=find_sources(os.path.join(*package.split('.'), name, "src")),
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["--expt-extended-lambda"],
        },
    )


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


VERSION_FILE = 'version.py'


def get_version():
    with open(VERSION_FILE, 'r') as f:
        exec(compile(f.read(), VERSION_FILE, 'exec'))
    return locals()['__version__']


def make_cuda_ext(name, module, sources):
    define_macros = []

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
    else:
        raise EnvironmentError('CUDA is required to compile MMDetection!')

    return CUDAExtension(name='{}.{}'.format(module, name),
                         sources=[os.path.join(*module.split('.'), p) for p in sources],
                         define_macros=define_macros,
                         extra_compile_args={
                             'cxx': [],
                             'nvcc': [
                                 '-D__CUDA_NO_HALF_OPERATORS__',
                                 '-D__CUDA_NO_HALF_CONVERSIONS__',
                                 '-D__CUDA_NO_HALF2_OPERATORS__',
                             ]
                         })


def parse_requirements(fname='requirements.txt', with_version=True):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == '__main__':
    setup(
        name='mmdet',
        version=get_version(),
        description='Open MMLab Detection Toolbox and Benchmark',
        # long_description=readme(),
        author='OpenMMLab',
        author_email='chenkaidev@gmail.com',
        keywords='computer vision, object detection',
        url='https://github.com/open-mmlab/mmdetection',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        package_data={'mmdet.ops': ['*/*.so']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        #        extras_require={
        #            'all': parse_requirements('requirements.txt'),
        #            'tests': parse_requirements('requirements/tests.txt'),
        #            'build': parse_requirements('requirements/build.txt'),
        #            'optional': parse_requirements('requirements/optional.txt'),
        #        },
        ext_modules=[
            make_cuda_ext(name='compiling_info', module='mmdet.ops.utils', sources=['src/compiling_info.cpp']),
            make_cuda_ext(name='nms_cpu', module='mmdet.ops.nms', sources=['src/nms_cpu.cpp']),
            make_cuda_ext(name='nms_cuda', module='mmdet.ops.nms', sources=['src/nms_cuda.cpp', 'src/nms_kernel.cu']),
            make_cuda_ext(name='roi_align_cuda',
                          module='mmdet.ops.roi_align',
                          sources=[
                              'src/roi_align_cuda.cpp',
                              'src/roi_align_kernel.cu',
                              'src/roi_align_kernel_v2.cu',
                          ]),
            make_cuda_ext(name='roi_pool_cuda',
                          module='mmdet.ops.roi_pool',
                          sources=['src/roi_pool_cuda.cpp', 'src/roi_pool_kernel.cu']),
            make_cuda_ext(name='deform_conv_cuda',
                          module='mmdet.ops.dcn',
                          sources=['src/deform_conv_cuda.cpp', 'src/deform_conv_cuda_kernel.cu']),
            make_cuda_ext(name='deform_pool_cuda',
                          module='mmdet.ops.dcn',
                          sources=['src/deform_pool_cuda.cpp', 'src/deform_pool_cuda_kernel.cu']),
            make_cuda_ext(name='sigmoid_focal_loss_cuda',
                          module='mmdet.ops.sigmoid_focal_loss',
                          sources=['src/sigmoid_focal_loss.cpp', 'src/sigmoid_focal_loss_cuda.cu']),
            make_cuda_ext(name='masked_conv2d_cuda',
                          module='mmdet.ops.masked_conv',
                          sources=['src/masked_conv2d_cuda.cpp', 'src/masked_conv2d_kernel.cu']),
            make_cuda_ext(name='affine_grid_cuda', module='mmdet.ops.affine_grid', sources=['src/affine_grid_cuda.cpp']),
            make_cuda_ext(name='grid_sampler_cuda',
                          module='mmdet.ops.grid_sampler',
                          sources=['src/cpu/grid_sampler_cpu.cpp', 'src/cuda/grid_sampler_cuda.cu', 'src/grid_sampler.cpp']),
            make_cuda_ext(name='carafe_cuda',
                          module='mmdet.ops.carafe',
                          sources=['src/carafe_cuda.cpp', 'src/carafe_cuda_kernel.cu']),
            make_cuda_ext(name='carafe_naive_cuda',
                          module='mmdet.ops.carafe',
                          sources=['src/carafe_naive_cuda.cpp', 'src/carafe_naive_cuda_kernel.cu']),
            make_extension('roi_sampling', 'mmdet.ops')
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)

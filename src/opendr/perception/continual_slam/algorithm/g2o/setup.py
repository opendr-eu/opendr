from setuptools import setup
from setuptools.command.install import install
from opendr.perception.continual_slam.algorithm.g2o.fix_g2o import main
from distutils.sysconfig import get_python_lib
import shutil
import glob
import subprocess

__library_file__ = 'g2opy/lib/g2o*.so'
__version__ = '0.0.1'

def build_g2o():
    subprocess.run(
        ["cd g2opy && mkdir build"],
        shell=True
    )
    run = subprocess.run(
        ["cd g2opy/build && cmake -DPYBIND11_PYTHON_VERSION=3.8 .."],
        shell=True,
        capture_output=True
    )
    print(run.stdout.decode('utf-8'))
    run = subprocess.run(
        ["cd g2opy/build && make -j4"],
        shell=True,
        capture_output=True
    )
    print(run.stdout.decode('utf-8'))

def copy_g2o():
    install_dir = get_python_lib()
    lib_file = glob.glob(__library_file__)
    assert len(lib_file) == 1     

    print('copying {} -> {}'.format(lib_file[0], install_dir))
    shutil.copy(lib_file[0], install_dir)

class InstallLocalPackage(install):
    def run(self):
        install.run(self)
        main()
        build_g2o()
        copy_g2o()

setup(
    name='g2opy',
    version=__version__,
    description='Python binding of C++ graph optimization framework g2o.',
    url='https://github.com/uoip/g2opy',
    license='BSD',
    cmdclass=dict(
        install=InstallLocalPackage
    ),
    keywords='g2o, SLAM, BA, ICP, optimization, python, binding',
    long_description="""This is a Python binding for c++ library g2o 
        (https://github.com/RainerKuemmerle/g2o).

        g2o is an open-source C++ framework for optimizing graph-based nonlinear 
        error functions. g2o has been designed to be easily extensible to a wide 
        range of problems and a new problem typically can be specified in a few 
        lines of code. The current implementation provides solutions to several 
        variants of SLAM and BA."""
)

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['opendr_example'],
    package_dir={'': 'scripts'})

setup(**setup_args)

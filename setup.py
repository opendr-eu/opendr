import os
from os.path import join
from setuptools import setup
from setuptools import find_packages
from Cython.Build import cythonize
import numpy
from configparser import ConfigParser

# Retrieve version
exec(open('src/opendr/_version.py').read())
try:
    __version__
except NameError:
    __version__ = '0.0'

# TODO: A common namespace should be defined (and still we will have conflicting files)...

# Read the long description
with open("description.txt") as f:
    long_description = f.read()

# Gather OpenDR submodules
opendr_modules = []
for base_dir in next(os.walk('src/opendr'))[1]:
    if base_dir not in ('engine', '__pycache__'):
        for x in next(os.walk('src/opendr/' + base_dir))[1]:
            if x not in ([], '__pycache__'):
                opendr_modules.append(join(base_dir, x))
opendr_modules.append('/engine')

# Add opendr-toolkit as meta package
opendr_modules.append('/opendr-toolkit')

for current_module in opendr_modules:
    module_name = current_module.split("/")[1]

    # Read the dependencies.ini
    try:
        parser = ConfigParser()
        parser.read(join("src/opendr", current_module, 'dependencies.ini'))
        dependencies = parser.get("runtime", "python").split('\n')
        dependencies = [x for x in dependencies if 'git' not in x]
    except Exception:
        dependencies = []

    if module_name == "opendr-toolkit":
        dependencies = []
        for x in opendr_modules:
            if x not in ("/opendr-toolkit", "/engine"):
                dependencies.append("opendr-toolkit-" + x.split("/")[1].replace("_", "-"))
        dependencies.append("opendr-toolkit-engine")

    # Gather all files
    data_files = []

    def traverse_dir(current_module):
        for root, dirs, files in os.walk(join("src/opendr", current_module)):
            for file in files:
                file_extension = file.split(".")[-1]
                # Add all files except from shared libraries
                if file_extension != "so" and file_extension != "py":
                    data_files.append(join(root.replace("src/opendr/", ""), file))

    traverse_dir(current_module)
    if module_name == 'engine':
        traverse_dir('utils')

    packages = find_packages(where="./src")
    if module_name == 'engine':
        packages = ['opendr.engine', 'opendr.utils']
    else:
        packages = [x for x in packages if module_name in x]
    name = "opendr-toolkit-" + module_name.replace("_", "-")
    print(len(data_files), len(packages))

    if module_name == 'opendr-toolkit':
        name = 'opendr-toolkit'
        description = 'Open Deep Learning Toolkit for Robotics '
    else:
        name = 'opendr-toolkit-' + module_name.replace("_", "-")
        description = 'Open Deep Learning Toolkit for Robotics (submodule: ' + current_module + ')'

    if current_module == 'perception/object_detection_2d':
        extra_params = {
            'ext_modules': cythonize(["src/opendr/perception/object_detection_2d/retinaface/algorithm/cython/*.pyx"]),
            'include_dirs': [numpy.get_include()]}
    else:
        extra_params = {}

    with open("MANIFEST.in", "w") as f:
        if module_name == 'opendr-toolkit':
            pass
        elif module_name == "engine":
            f.write("recursive-include src/opendr/engine *\n")
            f.write("include src/opendr/engine *\n")
            f.write("include src/opendr/utils *\n")
        else:
            f.write("recursive-include src/opendr/" + current_module + " *\n")
            f.write("include src/opendr/" + current_module.split("/")[0] + " *\n")
        f.write("include description.txt\n")

    setup(
        name=name,
        version=__version__,
        description=description,
        long_description=long_description,
        author='OpenDR consortium',
        author_email='tefas@csd.auth.gr',
        packages=packages,
        url='https://github.com/opendr-eu/opendr',
        license='LICENSE',
        package_dir={"": "src"},
        install_requires=dependencies,
        # package_data={'': data_files},
        **extra_params
    )

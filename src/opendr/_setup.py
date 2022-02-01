# Copyright 2020-2022 OpenDR European Project
#
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

from setuptools import setup
import os
from os.path import join
from configparser import ConfigParser
from setuptools import find_packages
from setuptools.command.install import install
import sys

author = 'OpenDR consortium'
author_email = 'tefas@csd.auth.gr'
url = 'https://github.com/opendr-eu/opendr'
license = 'LICENSE'
# Retrieve version
exec(open('src/opendr/_version.py').read())
try:
    __version__
except NameError:
    __version__ = '0.0'

# Read the long description
with open("description.txt") as f:
    long_description = f.read()

os.environ['DISABLE_BCOLZ_AVX2'] = 'true'


def get_packages(module=None):
    packages = find_packages(where="./src")
    if module:
        module_short_name = module
        if module != 'engine':
            module_short_name = module.split("/")[1]
        packages = [x for x in packages if module_short_name in x]
        name = "opendr-toolkit-" + module_short_name.replace("_", "-")
    else:
        name = "opendr-toolkit"

    if module == 'engine':
        packages.append('opendr.utils')
    packages.append('opendr.perception')
    packages.append('opendr.engine')
    packages.append('opendr.control')
    packages.append('opendr.planning')
    packages.append('opendr.simulation')
    packages.append('opendr')
    return name, packages


def generate_manifest(module=None):
    with open("MANIFEST.in", "w") as f:
        if module == "engine":
            f.write("recursive-include src/opendr/engine *\n")
            f.write("include src/opendr/engine *\n")
            f.write("include src/opendr/utils *\n")
        else:
            f.write("recursive-include " + join("src/opendr", module) + " *\n")
            f.write("include " + join("src/opendr", module.split("/")[0]) + " *\n")
            f.write("exclude " + join("src/opendr", module.split("/")[0]) + "/__init__.py \n")

        f.write("exclude src/opendr/__init__.py \n")
        f.write("include description.txt \n")
        f.write("include README.md \n")
        f.write("include src/opendr/_version.py \n")
        f.write("include src/opendr/_setup.py \n")

def get_description(module=None):
    if module:
        return 'Open Deep Learning Toolkit for Robotics (submodule: ' + module + ')'
    else:
        return 'Open Deep Learning Toolkit for Robotics'


def get_dependencies(current_module):
    # Read all the dependencies.ini for each tool category
    dependencies = []
    skipped_dependencies = []
    # Get all subfolders
    paths = ['.']

    for file in os.listdir(join("src/opendr", current_module)):
        if os.path.isdir(join("src/opendr", current_module, file)):
            paths.append(file)

    parser = ConfigParser()
    for path in paths:
        try:
            parser.read(join("src/opendr", current_module, path, 'dependencies.ini'))
            cur_deps = parser.get("runtime", "python").split('\n')
        except Exception:
            cur_deps = []
        # Add dependencies found (filter git-based ones and local ones)
        for x in cur_deps:
            if 'git' in x or '$' in x:
                skipped_dependencies.append(x)
            else:
                dependencies.append(x)

    dependencies = list(set(dependencies))
    print("deps = ", dependencies)
    skipped_dependencies = list(set(skipped_dependencies))
    return dependencies, skipped_dependencies


def build_package(module):

    if module == 'perception/object_detection_2d':
        from Cython.Build import cythonize
        import numpy
        extra_params = {
            'ext_modules':
                cythonize([join("src/opendr/perception/object_detection_2d/retinaface/algorithm/cython/*.pyx")]),
            'include_dirs': [numpy.get_include()]}
    else:
        extra_params = {}

    name, packages = get_packages(module)
    dependencies, skipped_dependencies = get_dependencies(module)
    generate_manifest(module)

    # Define class for post installation scripts
    class PostInstallScripts(install):
        def run(self):
            install.run(self)
            import subprocess

            # Install potential git repos during post installation
            for package in skipped_dependencies:
                if 'git' in package:
                    subprocess.call([sys.executable, '-m', 'pip', 'install', package])

    print("-------")
    print(packages, dependencies)
    print("-------")
    setup(
        name=name,
        version=__version__,
        description=get_description(module),
        long_description=long_description,
        author=author,
        author_email=author_email,
        packages=packages,
        url=url,
        license=license,
        package_dir={"": "./src"},
        install_requires=dependencies,
        cmdclass={
            'develop': PostInstallScripts,
           'install': PostInstallScripts,
        },
        **extra_params
    )
    print("----")
    print(skipped_dependencies)
    print("----")



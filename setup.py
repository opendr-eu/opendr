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
# TODO: Add engine as base dependecy
# TODO: Dependencies for pip generation: Cython numpy

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
opendr_modules.append('engine')

# Keep track of dependencies that have been skipped
tools_with_skipped_dependencies = {}
tools_list = []
print("------")
print(opendr_modules)
print("------")
for current_module in opendr_modules:

    if '/' in current_module:
        module_short_name = current_module.split("/")[1]
    else:
        module_short_name = current_module

    # Read all the dependencies.ini for each tool category
    dependencies = []
    # Get all subfolders
    paths = ['.']
    print("hi there")
    print(current_module)
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
                if module_short_name in tools_with_skipped_dependencies:
                    tools_with_skipped_dependencies[module_short_name].append(x)
                else:
                    tools_with_skipped_dependencies[module_short_name] = [x]
            else:
                dependencies.append(x)

    # Remove duplicates
    dependencies = list(set(dependencies))
    if module_short_name in tools_with_skipped_dependencies:
        tools_with_skipped_dependencies[module_short_name] = list(
            set(tools_with_skipped_dependencies[module_short_name]))

    packages = find_packages(where="./src")
    if module_short_name == 'engine':
        packages = ['opendr.engine', 'opendr.utils']
        name = "opendr-toolkit-engine"
    else:
        packages = [x for x in packages if module_short_name in x]
        name = "opendr-toolkit-" + module_short_name.replace("_", "-")
    # Keep track of the packages generated
    tools_list.append(name)

    name = 'opendr-toolkit-' + module_short_name.replace("_", "-")
    description = 'Open Deep Learning Toolkit for Robotics (submodule: ' + current_module + ')'

    if current_module == 'perception/object_detection_2d':
        extra_params = {
            'ext_modules': cythonize(["src/opendr/perception/object_detection_2d/retinaface/algorithm/cython/*.pyx"]),
            'include_dirs': [numpy.get_include()]}
    else:
        extra_params = {}

    with open("MANIFEST.in", "w") as f:
        if module_short_name == 'opendr-toolkit':
            pass
        elif module_short_name == "engine":
            f.write("recursive-include src/opendr/engine *\n")
            f.write("include src/opendr/engine *\n")
            f.write("include src/opendr/utils *\n")
        else:
            f.write("recursive-include src/opendr/" + current_module + " *\n")
            f.write("include src/opendr/" + current_module.split("/")[0] + " *\n")

        f.write("include description.txt\n")
        f.write("include src/opendr/_version.py\n")


    if name =='opendr-toolkit-hyperparameter-tuner':
        print(name, __version__, dependencies)
        print(packages)
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
            **extra_params
        )

    # Remove MANIFEST.in
    os.remove("MANIFEST.in")


# TODO: Create metapackage
# TODO: Check what to do with git/skipped dependecies
# TODO: Add manual dependecies to pip
# Build opendr-toolkit meta package
# # Add opendr-toolkit as meta package
# opendr_modules.append('/opendr-toolkit')
#
#    if module_short_name == "opendr-toolkit":
#         dependencies = []
#         for x in opendr_modules:
#             if x not in ("/opendr-toolkit", "/engine"):
#                 dependencies.append("opendr-toolkit-" + x.split("/")[1].replace("_", "-"))
#         dependencies.append("opendr-toolkit-engine")
#
# if module_short_name == 'opendr-toolkit':
#     name = 'opendr-toolkit'
#     description = 'Open Deep Learning Toolkit for Robotics '

for x in tools_with_skipped_dependencies:
    print(x, "->")
    print(tools_with_skipped_dependencies[x])


import os
from pip._internal import main as pip
from configparser import ConfigParser

dependency_file = "python_dependencies.txt"

def read_ini(path):
    parser = ConfigParser()
    parser.read(path)
    python_dependencies = parser.get('runtime', 'python')
    if python_dependencies:
        for package in python_dependencies.split():
             f = open(dependency_file, "a")
             f.write(package + '\n')


# Clear dependencies
open(dependency_file, 'w').close()
# Extract generic dependencies
read_ini('dependencies.ini')
# Loop through tools and extract dependencies
opendr_home = os.environ.get('OPENDR_HOME') 
for subdir, dirs, files in os.walk(opendr_home + '/src'):
    for filename in files:
        if filename == 'dependencies.ini':
            read_ini(subdir + os.sep + filename)    


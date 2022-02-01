from setuptools import setup
import os
from os.path import join

# Import tools for setup
exec(open(join(os.environ['OPENDR_HOME'], 'src/opendr/_setup.py')).read())
build_package("perception/speech_recognition")

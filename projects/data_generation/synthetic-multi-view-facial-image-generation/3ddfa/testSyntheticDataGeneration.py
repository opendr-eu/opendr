# Copyright 1996-2020 OpenDR European Project
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

# MIT License
#
# Copyright (c) 2019 Jian Zhao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import path_helper
import argparse
from SyntheticDataGeneration import MultiviewDataGenerationLearner
__all__ = ['path_helper']

parser = argparse.ArgumentParser()
parser.add_argument('-path_in', default='/home/user/Pictures/TEST', type=str, help='Give the path of image folder')
parser.add_argument('-path_3ddfa', default='./', type=str, help='Give the path of 3ddfa folder')
parser.add_argument('-save_path', default='./results', type=str, help='Give the path of results folder')
parser.add_argument('-val_yaw',  default="10,20", nargs='+', type=str, help='yaw poses list between [-90,90] ')
parser.add_argument('-val_pitch', default="30,40", nargs='+', type=str,  help='pitch poses list between [-90,90] ')
args = parser.parse_args()
synthetic = MultiviewDataGenerationLearner(path_in=args.path_in, path_3ddfa=args.path_3ddfa, save_path=args.save_path,
                                           val_yaw=args.val_yaw, val_pitch=args.val_pitch)
if __name__ == '__main__':
   synthetic.eval()

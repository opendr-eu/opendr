#!/usr/bin/env python3
# pylint: disable = line-too-long

import os
from pathlib import Path

INPUT_FILE = Path(__file__).parent / 'g2opy' / 'python' / 'core' / 'eigen_types.h'
assert INPUT_FILE.exists()

tmp_file = Path(str(INPUT_FILE) + '_tmp')

with open(INPUT_FILE, 'r', encoding='utf-8') as input_file:
    with open(tmp_file, 'w', encoding='utf-8') as output_file:
        for i, line in enumerate(input_file.readlines()):
            if line == '        .def("x", (double (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::x)\n':
                line = '        .def("x", (double &(Eigen::Quaterniond::*)()) & Eigen::Quaterniond::x)\n'
            elif line == '        .def("y", (double (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::y)\n':
                line = '        .def("y", (double &(Eigen::Quaterniond::*)()) & Eigen::Quaterniond::y)\n'
            elif line == '        .def("z", (double (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::z)\n':
                line = '        .def("z", (double &(Eigen::Quaterniond::*)()) & Eigen::Quaterniond::z)\n'
            elif line == '        .def("w", (double (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::w)\n':
                line = '        .def("w", (double &(Eigen::Quaterniond::*)()) & Eigen::Quaterniond::w)\n'
            output_file.write(line)

os.rename(tmp_file, INPUT_FILE)
/*****************************************************************
 *
 * This file is part of the GMAPPING project
 *
 * GMAPPING Copyright (c) 2004 Giorgio Grisetti,
 * Cyrill Stachniss, and Wolfram Burgard
 *
 * This software is licensed under the "Creative Commons
 * License (Attribution-NonCommercial-ShareAlike 2.0)"
 * and is copyrighted by Giorgio Grisetti, Cyrill Stachniss,
 * and Wolfram Burgard.
 *
 * Further information on this license can be found at:
 * http://creativecommons.org/licenses/by-nc-sa/2.0/
 *
 * GMAPPING is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.
 *
 *****************************************************************/

// Copyright 2020-2023 OpenDR European Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gmapping/configfile/configfile.h"
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace GMapping;

int main(int argc, char **argv) {
  if (argc != 2) {
    cerr << "Usage:  " << argv[0] << " [initifle]" << endl;
    exit(0);
  }

  ConfigFile cfg;
  cfg.read(argv[argc - 1]);

  cout << "-- values from configfile --" << endl;
  cfg.dumpValues(cout);

  cout << "-- adding a value --" << endl;
  cfg.value("unkown", "unkown", std::string("the new value!"));

  cout << "-- values from configfile & added values --" << endl;
  cfg.dumpValues(cout);

  if (((std::string)cfg.value("unkown", "unkown", std::string("the new value!"))) != std::string("the new value!"))
    cerr << "strange error, check strings" << endl;

  return 0;
}

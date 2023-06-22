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

/*
 * Copyright 2020-2023 OpenDR European Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef COMMANDLINE_H
#define COMMANDLINE_H

#define parseFlag(name, value)     \
  if (!strcmp(argv[c], name)) {    \
    value = true;                  \
    cout << name << " on" << endl; \
    recognized = true;             \
  }

#define parseString(name, value)                \
  if (!strcmp(argv[c], name) && c < argc - 1) { \
    c++;                                        \
    value = argv[c];                            \
    cout << name << "=" << value << endl;       \
    recognized = true;                          \
  }

#define parseDouble(name, value)                \
  if (!strcmp(argv[c], name) && c < argc - 1) { \
    c++;                                        \
    value = atof(argv[c]);                      \
    cout << name << "=" << value << endl;       \
    recognized = true;                          \
  }

#define parseInt(name, value)                   \
  if (!strcmp(argv[c], name) && c < argc - 1) { \
    c++;                                        \
    value = atoi(argv[c]);                      \
    cout << name << "=" << value << endl;       \
    recognized = true;                          \
  }

#define CMD_PARSE_BEGIN(i, count) \
  {                               \
    int c = i;                    \
    while (c < count) {           \
      bool recognized = false;

#define CMD_PARSE_END                                                           \
  if (!recognized)                                                              \
    cout << "COMMAND LINE: parameter " << argv[c] << " not recognized" << endl; \
  c++;                                                                          \
  }                                                                             \
  }

#define CMD_PARSE_BEGIN_SILENT(i, count) \
  {                                      \
    int c = i;                           \
    while (c < count) {                  \
      bool recognized = false;

#define CMD_PARSE_END_SILENT \
  c++;                       \
  }                          \
  }

#define parseFlagSilent(name, value) \
  if (!strcmp(argv[c], name)) {      \
    value = true;                    \
    recognized = true;               \
  }

#define parseStringSilent(name, value)          \
  if (!strcmp(argv[c], name) && c < argc - 1) { \
    c++;                                        \
    value = argv[c];                            \
    recognized = true;                          \
  }

#define parseDoubleSilent(name, value)          \
  if (!strcmp(argv[c], name) && c < argc - 1) { \
    c++;                                        \
    value = atof(argv[c]);                      \
    recognized = true;                          \
  }

#define parseIntSilent(name, value)             \
  if (!strcmp(argv[c], name) && c < argc - 1) { \
    c++;                                        \
    value = atoi(argv[c]);                      \
    recognized = true;                          \
  }

#endif

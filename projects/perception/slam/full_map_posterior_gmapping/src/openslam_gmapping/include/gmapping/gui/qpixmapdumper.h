
/*
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

#ifndef _QPIXMAPDUMPER_H_
#define _QPIXMAPDUMPER_H_

#include <qimage.h>
#include <qpixmap.h>
#include <string>

struct QPixmapDumper {
  QPixmapDumper(std::string prefix, int cycles);

  void reset();

  std::string prefix;
  std::string format;

  bool dump(const QPixmap &pixmap);

  int counter;
  int cycles;
  int frame;
};

#endif

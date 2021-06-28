
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

#ifndef _QNAVIGATOR_WIDGET_H
#define _QNAVIGATOR_WIDGET_H

#include <gmapping/utils/point.h>
#include <list>
#include "gmapping/gui/qmappainter.h"
#include "gmapping/gui/qpixmapdumper.h"

class QNavigatorWidget : public QMapPainter {
public:
  QNavigatorWidget(QWidget *parent = 0, const char *name = 0, WFlags f = 0);

  virtual ~QNavigatorWidget();

  std::list<GMapping::IntPoint> trajectoryPoints;
  bool repositionRobot;
  GMapping::IntPoint robotPose;
  double robotHeading;
  bool confirmLocalization;
  bool enableMotion;
  bool startWalker;
  bool startGlobalLocalization;
  bool trajectorySent;
  bool goHome;
  bool wantsQuit;
  bool writeImages;
  QPixmapDumper dumper;
  bool drawRobot;

protected:
  virtual void paintEvent(QPaintEvent *paintevent);

  virtual void mousePressEvent(QMouseEvent *e);

  virtual void keyPressEvent(QKeyEvent *e);
};

#endif

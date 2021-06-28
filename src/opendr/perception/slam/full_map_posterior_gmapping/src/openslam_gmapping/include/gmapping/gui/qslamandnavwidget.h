
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

#ifndef _QSLAMANDNAV_WIDGET_H
#define _QSLAMANDNAV_WIDGET_H

#include <gmapping/utils/point.h>
#include <list>
#include "gmapping/gui/qmappainter.h"
#include "gmapping/gui/qpixmapdumper.h"

class QSLAMandNavWidget : public QMapPainter {
public:
  QSLAMandNavWidget(QWidget *parent = 0, const char *name = 0, WFlags f = 0);

  virtual ~QSLAMandNavWidget();

  std::list<GMapping::IntPoint> trajectoryPoints;
  GMapping::IntPoint robotPose;
  double robotHeading;

  bool slamRestart;
  bool slamFinished;
  bool enableMotion;
  bool startWalker;
  bool trajectorySent;
  bool goHome;
  bool wantsQuit;
  bool printHelp;
  bool saveGoalPoints;
  bool writeImages;
  bool drawRobot;
  QPixmapDumper dumper;

protected:
  virtual void paintEvent(QPaintEvent *paintevent);

  virtual void mousePressEvent(QMouseEvent *e);

  virtual void keyPressEvent(QKeyEvent *e);
};

#endif

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

#ifndef QGRAPHPAINTER_H
#define QGRAPHPAINTER_H

#include <gmapping/utils/gvalues.h>
#include <qpainter.h>
#include <qpixmap.h>
#include <qstring.h>
#include <qwidget.h>
#include <qwmatrix.h>
#include <deque>

typedef std::deque<double> DoubleDeque;

class QGraphPainter : public QWidget {
  Q_OBJECT
public:
  QGraphPainter(QWidget *parent = 0, const char *name = 0, WFlags f = 0);

  virtual ~QGraphPainter();

public slots:
  void

    clear();

  void valueAdded(double);

  void valueAdded(double, double, double);

  void setYReference(double y);

  void disableYReference();

  void setRange(double min, double max);

  void start(int period);

  void setTitle(const char *title);

  void setAutoscale(bool a);

  bool getAutoscale() const;

protected:
  virtual void timerEvent(QTimerEvent *te);

  virtual void resizeEvent(QResizeEvent *);

  double min, max, reference;
  DoubleDeque values;
  bool autoscale;
  bool m_useYReference;
  int timer;

  virtual void paintEvent(QPaintEvent *paintevent);

  QPixmap *m_pixmap;
  QString title;
};

#endif

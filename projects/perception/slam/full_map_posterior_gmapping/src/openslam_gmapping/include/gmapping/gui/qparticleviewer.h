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

#ifndef QPARTICLEVIEWER_H
#define QPARTICLEVIEWER_H

#include <assert.h>
#include <qimage.h>
#include <qpainter.h>
#include <qpixmap.h>
#include <qtextstream.h>
#include <qwidget.h>
#include <qwmatrix.h>
#include <iostream>
#include <sstream>
#include <vector>

#include <gmapping/utils/point.h>
#include "gmapping/gui/gsp_thread.h"

namespace GMapping {

  class QParticleViewer : public QWidget {
    Q_OBJECT
  public:
    struct StartParameters {
      // motionmodel
      double srr, srt, str, stt;
      // map
      double xmin, ymin, xmax, ymax, delta;
      OrientedPoint initialPose;
      // likelihood
      double lsigma, lgain;
      unsigned int lskip;
      // update
      double linearUpdate, angularUpdate;
      // filter
      unsigned int particles;
      double resampleThreshold;
      // mode
      bool drawFromObservation;
      // output
      const char *outFileName;
    };

    struct MatchingParameters {
      // ranges
      double maxrange, urange;
      // score
      double ssigma, sreg, scrit;
      unsigned int ksize;
      // search
      double lstep, astep;
      unsigned int iterations;
    };

    void refreshParameters();  // reads the parameters from the thread
    inline void setGSP(GridSlamProcessorThread *thread) { gfs_thread = thread; }

    typedef std::vector<OrientedPoint> OrientedPointVector;

    QParticleViewer(QWidget *parent = 0, const char *name = 0, WFlags f = 0, GridSlamProcessorThread *thread = 0);

    virtual ~QParticleViewer();

    virtual void timerEvent(QTimerEvent *te);

    virtual void resizeEvent(QResizeEvent *);

    void drawFromFile();

    void drawFromMemory();

    void drawMap(const ScanMatcherMap &map);

    void start(int period);

    QTextIStream *tis;

    MatchingParameters matchingParameters;
    StartParameters startParameters;

    int writeToFile;
  public slots:
    void

      setMatchingParameters(const MatchingParameters &mp);

    void setStartParameters(const StartParameters &mp);

    void start();

    void stop();

    void loadFile(const char *);

  signals:
    void

      neffChanged(double);

    void poseEntropyChanged(double, double, double);

    void trajectoryEntropyChanged(double, double, double);

    void mapsEntropyChanged(double);

    void mapsIGainChanged(double);

  protected:
    ifstream inputStream;
    ofstream outputStream;

  protected:
    inline Point pic2map(const IntPoint &p) { return viewCenter + Point(p.x / mapscale, -p.y / mapscale); }

    inline IntPoint map2pic(const Point &p) {
      return IntPoint((int)((p.x - viewCenter.x) * mapscale), (int)((viewCenter.y - p.y) * mapscale));
    }

    int timer;

    virtual void paintEvent(QPaintEvent *paintevent);

    void drawParticleMove(const OrientedPointVector &start, const OrientedPointVector &end);

    QPixmap *m_pixmap;

    // thread interaction
    GridSlamProcessorThread *gfs_thread;
    GridSlamProcessorThread::EventDeque history;

    // mouse movement
    virtual void mousePressEvent(QMouseEvent *);

    virtual void mouseReleaseEvent(QMouseEvent *);

    virtual void mouseMoveEvent(QMouseEvent *);

    QPoint draggingPos;
    bool dragging;

    // particle plotting
    virtual void keyPressEvent(QKeyEvent *e);

    // map painting
    double mapscale;
    Point viewCenter;
    Point bestParticlePose;
    ScanMatcherMap *bestMap;

    // view mode
    bool showPaths;
    bool showBestPath;

    // file plotting
    QParticleViewer::OrientedPointVector m_oldPose, m_newPose;
    unsigned int m_particleSize;
    bool m_refresh;
    int count;
  };

};  // namespace GMapping

#endif

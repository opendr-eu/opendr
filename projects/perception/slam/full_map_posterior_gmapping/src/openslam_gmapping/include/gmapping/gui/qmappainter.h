
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

#ifndef QMAPPAINTER_H
#define QMAPPAINTER_H

#include <gmapping/utils/gvalues.h>
#include <gmapping/utils/point.h>
#include <qpainter.h>
#include <qpixmap.h>
#include <qstring.h>
#include <qwidget.h>

class QMapPainter : public QWidget {
public:
  QMapPainter(QWidget *parent = 0, const char *name = 0, WFlags f = 0);

  virtual ~QMapPainter();

public:
  template<typename Cell> void setPixmap(unsigned int xsize, unsigned int ysize, Cell **values);

  template<typename Iterator>
  void drawPoints(const Iterator &begin, const Iterator &end, unsigned char r, unsigned char g, unsigned char b);

  void start(int period);

protected:
  virtual void timerEvent(QTimerEvent *te);

  virtual void resizeEvent(QResizeEvent *);

  int timer;

  virtual void paintEvent(QPaintEvent *paintevent);

  QPixmap *m_pixmap;
};

template<typename Cell> void QMapPainter::setPixmap(unsigned int xsize, unsigned int ysize, Cell **values) {
  QSize s(xsize, ysize);
  m_pixmap->resize(s);
  m_pixmap->fill(Qt::white);
  QPainter painter(m_pixmap);
  for (unsigned int x = 0; x < (unsigned int)xsize; x++)
    for (unsigned int y = 0; y < (unsigned int)ysize; y++) {
      double v = (double)values[x][y];

      if (v >= 0) {
        unsigned int grayVal = (unsigned char)(255 - (unsigned char)(255 * v));
        painter.setPen(QColor(grayVal, grayVal, grayVal));
      } else {
        painter.setPen(QColor(255, 100, 100));
      }
      painter.drawPoint(x, ysize - y);
    }
}

template<typename Iterator>
void QMapPainter::drawPoints(const Iterator &begin, const Iterator &end, unsigned char r, unsigned char g, unsigned char b) {
  QPainter painter(m_pixmap);
  painter.setPen(QColor(r, g, b));
  for (Iterator it = begin; it != end; it++) {
    GMapping::IntPoint p = (GMapping::IntPoint)*it;
    painter.drawPoint(p.x, height() - p.y);
  }
}

#endif

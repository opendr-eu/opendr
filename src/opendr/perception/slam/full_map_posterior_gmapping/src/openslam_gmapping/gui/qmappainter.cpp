
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

#include "gmapping/gui/qmappainter.h"
#include "moc_qmappainter.cpp"

QMapPainter::QMapPainter(QWidget *parent, const char *name, WFlags f) :
  QWidget(parent, name, f | WRepaintNoErase | WResizeNoErase) {
  m_pixmap = new QPixmap(size());
  m_pixmap->fill(Qt::white);
}

void QMapPainter::resizeEvent(QResizeEvent *sizeev) {
  m_pixmap->resize(sizeev->size());
}

QMapPainter::~QMapPainter() {
  delete m_pixmap;
}

void QMapPainter::timerEvent(QTimerEvent *te) {
  if (te->timerId() == timer)
    update();
}

void QMapPainter::start(int period) {
  timer = startTimer(period);
}

void QMapPainter::paintEvent(QPaintEvent *) {
  bitBlt(this, 0, 0, m_pixmap, 0, 0, m_pixmap->width(), m_pixmap->height(), CopyROP);
}

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


#include <unistd.h>
#include "gmapping/gui/gsp_thread.h"

using namespace GMapping;

int main(int argc, char **argv) {
    cerr << "GMAPPING copyright 2004 by Giorgio Grisetti, Cyrill Stachniss," << endl;
    cerr << "and Wolfram Burgard. To be published under the CreativeCommons license," << endl;
    cerr << "see: http://creativecommons.org/licenses/by-nc-sa/2.0/" << endl << endl;


    GridSlamProcessorThread *gsp = new GridSlamProcessorThread;
    if (gsp->init(argc, argv)) {
        cout << "GSP INIT ERROR" << endl;
        return -1;
    }
    cout << "GSP INITIALIZED" << endl;
    if (gsp->loadFiles()) {
        cout << "GSP READFILE ERROR" << endl;
        return -2;
    }
    cout << "FILES LOADED" << endl;
    gsp->setMapUpdateTime(1000000);
    gsp->start();
    cout << "THREAD STARTED" << endl;
    bool done = false;
    while (!done) {
        GridSlamProcessorThread::EventDeque events = gsp->getEvents();
        for (GridSlamProcessorThread::EventDeque::iterator it = events.begin(); it != events.end(); it++) {
            cout << flush;
            GridSlamProcessorThread::DoneEvent *doneEvent = dynamic_cast<GridSlamProcessorThread::DoneEvent *>(*it);
            if (doneEvent) {
                done = true;
                cout << "DONE!" << endl;
                gsp->stop();
            }
            if (*it)
                delete (*it);
        }
    }
}

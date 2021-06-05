#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>


using namespace std;

ostream &printpgm(ostream &os, int xsize, int ysize, const double *const *matrix) {
    if (!os)
        return os;
    os << "P5" << endl << xsize << endl << ysize << endl << 255 << endl;
    for (int y = ysize - 1; y >= 0; y--) {
        for (int x = 0; x < xsize; x++) {
            unsigned char c = (unsigned char) (255 * fabs(1. - matrix[x][y]));
            os.put(c);
        }
    }
    return os;
}

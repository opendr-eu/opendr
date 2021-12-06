#include <stdbool.h>

typedef struct SmplSkel {
  int ids[25];
  char joints[25][100];
  double duration;
  double **orientation;
  double **translation;
  int numFrames;
}SmplSkel;

void initialize_skel(SmplSkel *smplSkel);
void read_smpl_orientation(const char *filename_orient, SmplSkel *smplSkel);
void read_smpl_translation(const char *filename_trans, SmplSkel *smplSkel);

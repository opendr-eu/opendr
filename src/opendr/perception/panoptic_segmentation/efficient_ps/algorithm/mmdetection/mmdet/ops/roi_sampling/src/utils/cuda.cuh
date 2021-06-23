#pragma once

/*
 * General settings and functions
 */
const int WARP_SIZE = 32;
const int MAX_BLOCK_SIZE = 1024;

static int getNumThreads(int nElem) {
  int threadSizes[6] = {32, 64, 128, 256, 512, MAX_BLOCK_SIZE};
  for (int i = 0; i < 6; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}
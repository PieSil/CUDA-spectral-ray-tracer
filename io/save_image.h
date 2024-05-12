#ifndef SPECTRAL_RT_PROJECT_SAVE_IMAGE_H
#define SPECTRAL_RT_PROJECT_SAVE_IMAGE_H

#include "CImg.h"

typedef unsigned int uint;

void save_img(unsigned char* r, unsigned char* g, unsigned char* b, const uint width, const uint height, const char* filename);

#endif
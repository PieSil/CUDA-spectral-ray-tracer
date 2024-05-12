
#ifndef SPECTRAL_RT_PROJECT_IMAGE_H
#define SPECTRAL_RT_PROJECT_IMAGE_H

#include "CImg.h"
#include <iostream>

typedef cimg_library::CImg<unsigned char> uchar_img;
typedef cimg_library::CImgDisplay img_display;

uchar_img get_image(unsigned char* r, unsigned char* g, unsigned char* b, const unsigned int width, const unsigned int height);

img_display get_image_display(const char* title);

void display_image(uchar_img& image, img_display& disp);

#endif
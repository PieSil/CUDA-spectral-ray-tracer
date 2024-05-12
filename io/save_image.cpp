#include "save_image.h"
#include <iostream>

void save_img(uchar_img& image, const char* filename) {
   

	image.save(filename);
	std::clog << "Image saved" << std::endl;

	/*
	while (!main_disp.is_closed()) {
		main_disp.wait();
	}
	*/
}
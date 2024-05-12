#include "save_image.h"
#include <iostream>

void save_img(unsigned char* r, unsigned char* g, unsigned char* b, const uint width, const uint height, const char* filename) {
    std::clog << "width: " << width << " height: " << height << std::endl;
	cimg_library::CImg<unsigned char> image(width, height, 1, 3);

    int index = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            // Set RGB values for the pixel
            image(x, y, 0, 0) = r[y*width + x]; // Red channel
            image(x, y, 0, 1) = g[y*width + x]; // Green channel
            image(x, y, 0, 2) = b[y*width + x]; // Blue channel
        }
    }

	image.save(filename);
	cimg_library::CImgDisplay main_disp;
    main_disp.set_title("Render");
    image.display(main_disp);
	std::clog << "Image saved" << std::endl;
	while (!main_disp.is_closed()) {
		main_disp.wait();
	}
}
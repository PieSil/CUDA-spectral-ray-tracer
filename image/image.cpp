#include "image.h"

uchar_img get_image(unsigned char* r, unsigned char* g, unsigned char* b, const unsigned int width, const unsigned int height) {
    uchar_img image(width, height, 1, 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            // Set RGB values for the pixel
            image(x, y, 0, 0) = r[y * width + x]; // Red channel
            image(x, y, 0, 1) = g[y * width + x]; // Green channel
            image(x, y, 0, 2) = b[y * width + x]; // Blue channel
        }
    }

    return image;

}

img_display get_image_display(const char* title) {
    cimg_library::CImgDisplay disp;
    disp.set_title(title);
    return disp;
}

void display_image(uchar_img& image, img_display& disp) {
    image.display(disp);
}
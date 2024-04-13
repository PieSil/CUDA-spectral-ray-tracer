//
// Created by pietr on 20/03/2024.
//

#ifndef SPECTRAL_RAY_TRACING_IO_CUH
#define SPECTRAL_RAY_TRACING_IO_CUH

#include "color.cuh"

inline void write_to_ppm(color* fb, int width, int height) {
    int num_pixels = width*height;
    // Output FB as Image
    cout << "P3\n" << width << " " << height << "\n255\n";
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j*width + i;

            /*
             * I don't know why values in the frame buffer get saved in an inverse order,
             * to fix this I just read the fb ina bottom-up fashion
             * if this comment is still here then I did not have enough time to identify the issue
             */
            color pixel_color = fb[(num_pixels-1) - pixel_index];
            write_color(cout, pixel_color);
        }
    }

    clog << "image saved"<< endl;
}

#endif //SPECTRAL_RAY_TRACING_IO_CUH

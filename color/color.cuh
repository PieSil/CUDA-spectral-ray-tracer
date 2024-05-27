//
// Created by pietr on 31/03/2024.
//

#ifndef COLOR_CONVERSION_COLOR_H
#define COLOR_CONVERSION_COLOR_H

#include "vec3.cuh"
#include "color_const.cuh"
#include "spectrum.cuh"

using color = vec3;

__host__ __device__
color sRGB_to_XYZ(color sRGB, const float *sRGB_to_XYZ_matrix);

__host__ __device__
color XYZ_to_sRGB(color xyz, const float *XYZ_to_sRGB_matrix);

__host__ __device__
float invert_channel_correction(float value);

__host__ __device__
float correct_channel(float value);

__host__ __device__
color expand_sRGB(color bounded_sRGB);

__host__
color spectrum_to_XYZ(const float* spectrum, float* power_distribution, int n_samples);

__device__
color dev_spectrum_to_XYZ(const float * const spectrum, const float * const power_distribution, const int n_samples);

__host__
void write_color(std::ostream &out, color pixel_color);

#endif //COLOR_CONVERSION_COLOR_H

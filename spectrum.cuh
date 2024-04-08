//
// Created by pietr on 31/03/2024.
//

#ifndef COLOR_CONVERSION_SPECTRUM_H
#define COLOR_CONVERSION_SPECTRUM_H

#include "cie_const.cuh"
#include "host_utility.cuh"
#include "cuda_utility.cuh"

__host__ __device__
float spectrum_interp(const float* spectrum, float lambda, int n_samples = N_CIE_SAMPLES);

__device__
void init_hero_wavelength(float *spectrum, uint n_lambdas, curandState *local_rand_state);

#endif //COLOR_CONVERSION_SPECTRUM_H

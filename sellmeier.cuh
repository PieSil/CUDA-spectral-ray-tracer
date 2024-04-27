#ifndef SPECTRAL_RT_PROJECT_SELLMEIER_CUH
#define SPECTRAL_RT_PROJECT_SELLMEIER_CUH

#include "utility.cuh"

const float BK7_b[3] = { 1.03961212f, 0.231792344f, 1.01046945f };
const float BK7_c[3] = { 6.00069867e-3f, 2.00179144e-2f , 1.03560653e2f };

__device__
float sellmeier_index(const float b[3], const float c[3], const float lambda);
#endif
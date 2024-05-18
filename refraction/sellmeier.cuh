#ifndef SPECTRAL_RT_PROJECT_SELLMEIER_CUH
#define SPECTRAL_RT_PROJECT_SELLMEIER_CUH

#include "utility.h"

const float BK7_b[3] = { 1.03961212f, 0.231792344f, 1.01046945f };
const float BK7_c[3] = { 6.00069867e-3f, 2.00179144e-2f , 1.03560653e2f };

const float fused_silica_b[3] = { 0.6961663f, 0.4079426f, 0.8974794f };
const float fused_silica_c[3] = { 0.0684043f, 0.1162414f , 9.896161f };

const float flint_glass_b[3] = { 1.34533359f, 0.209073176f, 0.937357162f };
const float flint_glass_c[3] = { 0.00997743871f, 0.0470450767f , 111.886764f };

__constant__ inline float dev_BK7_b[3];
__constant__ inline float dev_BK7_c[3];
__constant__ inline float dev_fused_silica_b[3];
__constant__ inline float dev_fused_silica_c[3];
__constant__ inline float dev_flint_glass_b[3];
__constant__ inline float dev_flint_glass_c[3];

__device__
float sellmeier_index(const float b[3], const float c[3], const float lambda);
#endif
//
// Created by pietr on 31/03/2024.
//

#ifndef COLOR_CONVERSION_CIE_CONST_H
#define COLOR_CONVERSION_CIE_CONST_H

#define N_CIE_SAMPLES 95
#define CIE_CURVE_RES 5.0f
#define CIE_Y_INTEGRAL 106.856895f
#define LAMBDA_MAX 830.0f
#define LAMBDA_MIN 360.0f

extern const float cie_x[N_CIE_SAMPLES];
extern const float cie_y[N_CIE_SAMPLES];
extern const float cie_z[N_CIE_SAMPLES];
extern const float normalized_cie_d65[N_CIE_SAMPLES];
extern const float cie_d65[N_CIE_SAMPLES];
__constant__ inline float dev_cie_x[N_CIE_SAMPLES];
__constant__ inline float dev_cie_y[N_CIE_SAMPLES];
__constant__ inline float dev_cie_z[N_CIE_SAMPLES];

#endif //COLOR_CONVERSION_CIE_CONST_H

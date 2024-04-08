//
// Created by pietr on 31/03/2024.
//

#ifndef COLOR_CONVERSION_COLOR_CONST_H
#define COLOR_CONVERSION_COLOR_CONST_H

extern const float d65_sRGB_to_XYZ[3][3];

extern const float d65_XYZ_to_sRGB[3][3];

extern const float d50_sRGB_to_XYZ[3][3];

extern const float d50_XYZ_to_sRGB[3][3];

__constant__ inline float dev_d65_sRGB_to_XYZ[3][3];

__constant__ inline float dev_d65_XYZ_to_sRGB[3][3];

#endif //COLOR_CONVERSION_COLOR_CONST_H

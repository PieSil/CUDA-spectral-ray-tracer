//
// Created by pietr on 31/03/2024.
//

#include "color_const.cuh"

/*
 * Matrices used to convert from sRGB to XYZ and vice-versa
 * d50 prefix indicates the Bradford-corrected matrices, unused for now
 * source: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
 */

const float d65_sRGB_to_XYZ[3][3] = {{0.4124564f, 0.3575761f, 0.1804375f},
                                            {0.2126729f, 0.7151522f, 0.0721750f},
                                            {0.0193339f, 0.1191920f, 0.9503041f}};

const float d65_XYZ_to_sRGB[3][3] = {{3.2404542f, -1.5371385f, -0.4985314f},
                                            {-0.9692660f, 1.8760108f, 0.0415560f},
                                            {0.0556434f, -0.2040259f, 1.0572252f}};

const float d50_sRGB_to_XYZ[3][3] = {{0.4360747f, 0.3850649f,  0.1430804f},
                                            {0.2225045f, 0.7168786f, 0.0606169f},
                                            {0.0139322f, 0.0971045f, 0.7141733f}};

const float d50_XYZ_to_sRGB[3][3] = {{3.1338561f, -1.6168667f,  -0.4906146f},
                                            {-0.9787684f, 1.9161415f, 0.0334540f},
                                            {0.0719453f, -0.2289914f, 1.4052427f}};


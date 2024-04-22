//
// Created by pietr on 31/03/2024.
//

#ifndef COLOR_CONVERSION_UTILITY_H
#define COLOR_CONVERSION_UTILITY_H

#include "std_libs.cuh"

#define PI 3.1415926535897932385f

using namespace std;

//Typedefs

typedef unsigned int uint;

//Constants

const float infinity = numeric_limits<float>::infinity();

//Utility Functions

__host__ __device__
inline float degrees_to_radians(float degrees) {
    return degrees * PI / 180.0f;
}

__host__
inline float random_float(float min, float max) {

    return ((float)rand() / RAND_MAX) * (max - min) + min;

}


#endif //COLOR_CONVERSION_UTILS_H
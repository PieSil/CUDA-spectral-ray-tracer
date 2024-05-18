//
// Created by pietr on 31/03/2024.
//

#ifndef COLOR_CONVERSION_UTILITY_H
#define COLOR_CONVERSION_UTILITY_H

#include "std_libs.h"

#define PI 3.1415926535897932385f

using namespace std;

//Typedefs

typedef unsigned int uint;

//Constants

const float infinity = numeric_limits<float>::infinity();

//Utility Functions

inline float random_float(float min, float max) {

    return ((float)rand() / RAND_MAX) * (max - min) + min;

}

inline void string_to_filename(string& str) {
    for (char& c : str) {
        if (c == ' ') {
            c = '_';
        }
        else {
            c = std::tolower(static_cast<unsigned char>(c));
        }
    }
}


#endif //COLOR_CONVERSION_UTILS_H

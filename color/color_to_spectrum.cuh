//
// Created by pietr on 01/04/2024.
//

#ifndef COLOR_CONVERSION_COLOR_TO_SPECTRUM_H
#define COLOR_CONVERSION_COLOR_TO_SPECTRUM_H

#include "color.cuh"
#include "srgb_to_spectrum.cuh"

/*
 * This file defines a series of functions needed in order to convert a color expressed in the sRGB color space
 * to the corresponding reflectance or illuminance spectrum, the illuminant is set to CIE D65
 * (no other color space or illuminant are supported as I'm writing this).
 * Since I didn't manage to find any other good resource, and spectrum-to-color conversion, while necessary,
 * is not the focus of this project, I decided to straight-up copy this code from the pbrt-v4 repository:
 * https://github.com/mmp/pbrt-v4
 *
 * NOTE: the color-to-spectrum conversion is very approximated, as I said, this was not the focus of the project.
 *       I also slightly altered the process of conversion from sRGB to illuminance spectrum, in order to
 *       better fit my needs.
 *
 * The specific file I used as reference is:
 * https://github.com/mmp/pbrt-v4/blob/39e01e61f8de07b99859df04b271a02a53d9aeb2/src/pbrt/cmd/rgb2spec_opt.cpp#L366
 * (I say reference since I kept only the fragments of code which where strictly necessary to me, but the right word is "copied")
 *
 * More details on the meaning of these constants and the reasoning behind the algorithms can be found here:
 * https://www.pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Color#fragment-RGBToSpectrumTablePublicConstants-1
 *
 * I very much thank the authors of pbrt, I couldn't have done this without them.
 *
 * Matt Pharr, Wenzel Jakob, and Greg Humphreys. 2016.
 * Physically Based Rendering: From Theory to Implementation (3rd ed.). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA.
 */

__host__ __device__
inline float sigmoid_inf_check(float x) {
    if (isinf(x)) return x > 0 ? 1 : 0;
    return 0.5f * x / std::sqrt(1.0f + x * x) + 0.5f;
}

template<typename T, typename U, typename V>
__host__ __device__
inline constexpr T Clamp(T val, U low, V high) {
    if (val < low) return T(low);
    else if (val > high) return T(high);
    else return val;
}

template<typename Predicate>
__host__ __device__
inline size_t FindInterval(size_t sz, const Predicate &pred) {
    using ssize_t = std::make_signed_t<size_t>;
    ssize_t size = (ssize_t) sz - 2, first = 1;
    while (size > 0) {
        size_t half = (size_t) size >> 1, middle = first + half;
        bool predResult = pred(middle);
        first = predResult ? middle + 1 : first;
        size = predResult ? size - (half + 1) : half;
    }
    return (size_t) Clamp((ssize_t) first - 1, 0, sz - 2);
}

__host__ __device__
inline float Lerp(float x, float a, float b) {
    return (1 - x) * a + x * b;
}

__host__
inline vec3 get_sigmoid_coeffs(color in_color, int res = 64) {
    float r = in_color.x();
    float g = in_color.y();
    float b = in_color.z();


    float rgb[3] = {r, g, b};

    if (r == g && g == b) {
        return vec3(0.0f, 0.0f, (r - .5f) / sqrt(r * (1 - r)));
    }

    int maxc = (r > g) ? ((r > b) ? 0 : 2) :
               ((g > b) ? 1 : 2);
    float z = rgb[maxc];
    float x = rgb[(maxc + 1) % 3] * (res - 1) / z;
    float y = rgb[(maxc + 2) % 3] * (res - 1) / z;

    int xi = std::min((int) x, res - 2), yi = std::min((int) y, res - 2),
            zi = FindInterval(res, [&](int i) { return sRGBToSpectrumTable_Scale[i] < z; });
    float dx = x - xi, dy = y - yi,
            dz =
            (z - sRGBToSpectrumTable_Scale[zi]) / (sRGBToSpectrumTable_Scale[zi + 1] - sRGBToSpectrumTable_Scale[zi]);

    float c[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 3; ++i) {
        auto co = [&](int dx, int dy, int dz) {
            return sRGBToSpectrumTable_Data[maxc][zi + dz][yi + dy][xi + dx][i];
        };

        c[i] = Lerp(dz, Lerp(dy, Lerp(dx, co(0, 0, 0), co(1, 0, 0)),
                             Lerp(dx, co(0, 1, 0), co(1, 1, 0))),
                    Lerp(dy, Lerp(dx, co(0, 0, 1), co(1, 0, 1)),
                         Lerp(dx, co(0, 1, 1), co(1, 1, 1))));
    }

    return vec3(c[2], c[1], c[0]);
}

__device__
inline vec3 dev_get_sigmoid_coeffs(color in_color, float* dev_sRGBToSpectrumTable_Data, int res = 64) {
    float r = in_color.x();
    float g = in_color.y();
    float b = in_color.z();


    float rgb[3] = {r, g, b};

    if (r == g && g == b) {
        return vec3(0.0f, 0.0f, (r - .5f) / sqrt(r * (1 - r)));
    }

    int maxc = (r > g) ? ((r > b) ? 0 : 2) :
               ((g > b) ? 1 : 2);
    float z = rgb[maxc];
    float x = rgb[(maxc + 1) % 3] * (res - 1) / z;
    float y = rgb[(maxc + 2) % 3] * (res - 1) / z;

    int xi = min((int) x, res - 2), yi = min((int) y, res - 2),
            zi = FindInterval(res, [&](int i) { return dev_sRGBToSpectrumTable_Scale[i] < z; });
    float dx = x - xi, dy = y - yi,
            dz =
            (z - dev_sRGBToSpectrumTable_Scale[zi]) / (dev_sRGBToSpectrumTable_Scale[zi + 1] - dev_sRGBToSpectrumTable_Scale[zi]);

    float c[3] = {0.0f, 0.0f, 0.0f};


    //int linear_index = int((((float(maxc)*3 + (float(zi)+dz))*64 + (float(yi)+dy))*64 + (float(xi)+dx))*64);
    for (int i = 0; i < 3; ++i) {
        int linear_index = (((maxc*64 + (zi+int(dz)))*64 + (yi+int(dy)))*64 + (xi+int(dx))) * 3 + i;
        auto co = [&](int dx, int dy, int dz) {
            return dev_sRGBToSpectrumTable_Data[linear_index];
        };

        c[i] = Lerp(dz, Lerp(dy, Lerp(dx, co(0, 0, 0), co(1, 0, 0)),
                             Lerp(dx, co(0, 1, 0), co(1, 1, 0))),
                    Lerp(dy, Lerp(dx, co(0, 0, 1), co(1, 0, 1)),
                         Lerp(dx, co(0, 1, 1), co(1, 1, 1))));
    }

    return vec3(c[2], c[1], c[0]);
}

__host__ __device__
inline float polynomial(float x, float c2, float c1, float c0) {
    return x * x * c2 + x * c1 + c0;
}

__host__
inline void srgb_to_illuminance_spectrum(color srgb_color, float *sampled_spectrum, float power = 1.0f) {
    float step = (LAMBDA_MAX - LAMBDA_MIN) / N_CIE_SAMPLES;
    vec3 coeffs = get_sigmoid_coeffs(srgb_color);
    float lambda = LAMBDA_MIN;

    for (int i = 0; i < N_CIE_SAMPLES; i++) {
        float x = polynomial(lambda, coeffs.z(), coeffs.y(), coeffs.x());
        float sigmoid_x = pow(power, 2.0f) * sigmoid_inf_check(x) * spectrum_interp(normalized_cie_d65, lambda);
        sampled_spectrum[i] = sigmoid_x;
        lambda += step;

    }
}

__device__
inline void dev_srgb_to_illuminance_spectrum(color srgb_color, float *sampled_spectrum, float power, float* dev_sRGBToSpectrumTable_Data) {
    float step = (LAMBDA_MAX - LAMBDA_MIN) / N_CIE_SAMPLES;
    vec3 coeffs = dev_get_sigmoid_coeffs(srgb_color, dev_sRGBToSpectrumTable_Data);
    float lambda = LAMBDA_MIN;

    for (int i = 0; i < N_CIE_SAMPLES; i++) {
        float x = polynomial(lambda, coeffs.z(), coeffs.y(), coeffs.x());
        float sigmoid_x = pow(power, 2.0f) * sigmoid_inf_check(x) * spectrum_interp(dev_normalized_cie_d65, lambda);
        sampled_spectrum[i] = sigmoid_x;
        lambda += step;

    }
}

__host__
inline void srgb_to_spectrum(color srgb_color, float *sampled_spectrum) {
    float step = (LAMBDA_MAX - LAMBDA_MIN) / N_CIE_SAMPLES;
    vec3 coeffs = get_sigmoid_coeffs(srgb_color);
    float lambda = LAMBDA_MIN;

    for (int i = 0; i < N_CIE_SAMPLES; i++) {

        float x = polynomial(lambda, coeffs.z(), coeffs.y(), coeffs.x());
        float sigmoid_x = sigmoid_inf_check(x);
        sampled_spectrum[i] = sigmoid_x;
        lambda += step;

    }
}

__device__
inline void dev_srgb_to_spectrum(color srgb_color, float *sampled_spectrum, float* dev_sRGBToSpectrumTable_Data) {
    float step = (LAMBDA_MAX - LAMBDA_MIN) / N_CIE_SAMPLES;
    vec3 coeffs = dev_get_sigmoid_coeffs(srgb_color, dev_sRGBToSpectrumTable_Data);
    float lambda = LAMBDA_MIN;

    for (int i = 0; i < N_CIE_SAMPLES; i++) {

        float x = polynomial(lambda, coeffs.z(), coeffs.y(), coeffs.x());
        float sigmoid_x = sigmoid_inf_check(x);
        sampled_spectrum[i] = sigmoid_x;

        lambda += step;

    }
}

#endif //COLOR_CONVERSION_COLOR_TO_SPECTRUM_H

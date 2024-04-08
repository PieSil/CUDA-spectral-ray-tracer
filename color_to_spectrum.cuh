//
// Created by pietr on 01/04/2024.
//

#ifndef COLOR_CONVERSION_COLOR_TO_SPECTRUM_H
#define COLOR_CONVERSION_COLOR_TO_SPECTRUM_H

#include "host_utility.cuh"
#include "color.cuh"
#include "srgb_to_spectrum.h"

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
    int linear_index = int((((float(maxc)*64 + (float(zi)+dz))*64 + (float(yi)+dy))*64 + (float(xi)+dx))*3);
    for (int i = 0; i < 3; ++i, ++linear_index) {
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

//__host__
//inline void srgb_to_illuminance_spectrum(color srgb_color, float* sampled_spectrum) {
//    float m = std::max({srgb_color.x(), srgb_color.y(), srgb_color.z()});
//    float scale = 1;
//    vec3 coeffs = get_sigmoid_coeffs(scale ? srgb_color/scale : color(0, 0, 0));
//    float lambda = LAMBDA_MIN;
//    for(int i = 0; i < N_CIE_SAMPLES; i++) {
//        float x = polynomial(lambda, coeffs.z(), coeffs.y(), coeffs.x());
//        //std::clog << "lambda: " << lambda << ", x: " << x << std::endl;
//        float sigmoid_x = scale * sigmoid_inf_check(x) * spectrum_interp(cie_d65, lambda);
//        printf("lambda: %f, value: %f\n", lambda, sigmoid_x);
//        sampled_spectrum[i] = sigmoid_x;
//        lambda += 5.0f;
//    }
//}
__host__
inline void srgb_to_illuminance_spectrum(color srgb_color, float *sampled_spectrum, float power = 1.0f) {
    float step = (LAMBDA_MAX - LAMBDA_MIN) / N_CIE_SAMPLES;
    vec3 coeffs = get_sigmoid_coeffs(srgb_color);
    float lambda = LAMBDA_MIN;

    for (int i = 0; i < N_CIE_SAMPLES; i++) {
        float x = polynomial(lambda, coeffs.z(), coeffs.y(), coeffs.x());
        float sigmoid_x = power*power * sigmoid_inf_check(x)  * spectrum_interp(normalized_cie_d65, lambda);
        printf("lambda: %f, value: %f\n", lambda, sigmoid_x);
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
        float sigmoid_x = power*power * sigmoid_inf_check(x)  * spectrum_interp(normalized_cie_d65, lambda);
        printf("lambda: %f, value: %f\n", lambda, sigmoid_x);
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
        printf("lambda: %f, value: %f\n", lambda, sigmoid_x);
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
        printf("lambda: %f, value: %f\n", lambda, sigmoid_x);
        sampled_spectrum[i] = sigmoid_x;
        lambda += step;

    }
}

#endif //COLOR_CONVERSION_COLOR_TO_SPECTRUM_H

//
// Created by pietr on 08/04/2024.
//

#ifndef SPECTRAL_RT_PROJECT_INTERVAL_CUH
#define SPECTRAL_RT_PROJECT_INTERVAL_CUH

#include "cuda_utility.cuh"

namespace interval {

    struct interval {
        __host__ __device__
        interval() : min(+FLT_MAX), max(-FLT_MAX) {} //Default interval is empty

        __host__ __device__
        interval(float _min, float _max) : min(_min), max(_max) {}

        __host__ __device__
        interval(const interval& a, const interval& b) : min(fmin(a.min, b.min)), max(fmax(a.max, b.max)) {}

        float min;
        float max;
    };

    __device__
    inline bool contains(const float min, const float max, const float x) {
        return min <= x && x <= max;
    }

    __device__
    bool surrounds(const float min, const float max, const float x) {
        return min < x && x < max;
    }

    __device__
    float dev_clamp(float x, const float min, const float max) {
        return device_clamp(x, min, max);
    }

    __device__
    interval expand(const float min, const float max, float delta) {
        auto padding = delta/2;
        return interval(min - padding, max + padding);
    }

    const static interval empty(+infinity, -infinity);

    const static interval universe(-infinity, + infinity);
}

#endif //SPECTRAL_RT_PROJECT_INTERVAL_CUH

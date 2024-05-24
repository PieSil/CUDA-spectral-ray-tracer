//
// Created by pietr on 08/04/2024.
//

#ifndef SPECTRAL_RT_PROJECT_INTERVAL_CUH
#define SPECTRAL_RT_PROJECT_INTERVAL_CUH

#include "cuda_utility.cuh"

namespace interval {

    struct numeric_interval {
        __host__ __device__
        numeric_interval() : min(+FLT_MAX), max(-FLT_MAX) {} //Default interval is empty

        __host__ __device__
        numeric_interval(float _min, float _max) : min(_min), max(_max) {}

        __host__ __device__
        numeric_interval(const numeric_interval& a, const numeric_interval& b) : min(fmin(a.min, b.min)), max(fmax(a.max, b.max)) {}

        __host__ __device__
        numeric_interval(const numeric_interval& other) : min(other.min), max(other.max) {

        }

        __host__ __device__
        numeric_interval& operator=(const numeric_interval& r) {
            min = r.min;
            max = r.max;
            return *this;
        }

        __host__ __device__
        const bool isEmpty() const {
            return min > max;
        }

        float min;
        float max;
    };

    __device__
    inline bool contains(const float min, const float max, const float x) {
        return min <= x && x <= max;
    }

    __device__
    inline bool surrounds(const float min, const float max, const float x) {
        return min < x && x < max;
    }

    __device__
    inline float dev_clamp(float x, const float min, const float max) {
        return device_clamp(x, min, max);
    }

    __host__ __device__
    inline numeric_interval expand(const float min, const float max, float delta) {
        auto padding = delta/2;
        return numeric_interval(min - padding, max + padding);
    }

    __host__ __device__
    inline float size(float min, float max) {
        return max - min;
    }

    inline const static numeric_interval empty(+infinity, -infinity);

    inline const static numeric_interval universe(-infinity, + infinity);
}

#endif //SPECTRAL_RT_PROJECT_INTERVAL_CUH

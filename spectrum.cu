//
// Created by pietr on 31/03/2024.
//

#include "spectrum.cuh"

/*
 * copied from pbrt-v4 repo (called cie-interp in the original code):
 * https://github.com/mmp/pbrt-v4/blob/39e01e61f8de07b99859df04b271a02a53d9aeb2/src/pbrt/cmd/rgb2spec_opt.cpp#L4
 */
__host__ __device__
float spectrum_interp(const float* spectrum, float lambda, int n_samples) {
    lambda -= LAMBDA_MIN;
    lambda *= (float(n_samples) - 1) / (LAMBDA_MAX - LAMBDA_MIN);
    int offset = (int) lambda;
    if (offset < 0)
        offset = 0;
    if (offset > n_samples - 2)
        offset = n_samples - 2;
    float weight = lambda - float(offset);
    return (1.0f - weight) * spectrum[offset] + weight * spectrum[offset + 1];
}

/**
 * @brief Initialized array of wavelengths and places hero wavelength at position 0.
 *
 * @param spectrum The destination array.
 * @param n_lambdas The number of wavelengths.
 * @param local_rand_state Random state for random number generation on device.
 */
__device__
void init_hero_wavelength(float *spectrum, uint n_lambdas, curandState *local_rand_state) {
    float step = (LAMBDA_MAX - LAMBDA_MIN) / float(n_lambdas);
    float hero = cuda_random_float(LAMBDA_MIN, LAMBDA_MAX, local_rand_state);
    spectrum[0] = hero;
    float lambda = hero;

    for (int i = 1; i < n_lambdas; i++) {
        lambda += step;
        if (lambda > LAMBDA_MAX) {
            float remainder = lambda - LAMBDA_MAX;
            lambda = LAMBDA_MIN + remainder;
        }
        spectrum[i] = lambda;
    }


}
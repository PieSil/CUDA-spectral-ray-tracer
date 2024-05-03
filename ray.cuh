//
// Created by pietr on 08/04/2024.
//

#ifndef SPECTRAL_RT_PROJECT_RAY_CUH
#define SPECTRAL_RT_PROJECT_RAY_CUH

#include "vec3.cuh"
#include "spectrum.cuh"
#include "cie_const.cuh"

#define N_RAY_WAVELENGTHS 7
#define RESOLUTION (LAMBDA_MAX-LAMBDA_MIN)/N_RAY_WAVELENGTHS;

    class ray {
    public:
        vec3 orig;
        vec3 dir;
        bool wavelengths_zeroed = false;
        float wavelengths[N_RAY_WAVELENGTHS] = {0.0f};
        float power_distr[N_RAY_WAVELENGTHS] = {0.0f};

        __device__ ray() {};

        __device__
        ray(const point3& origin, const vec3& direction, curandState* local_rand_state) : orig(origin), dir(direction) {
            init_spectrum(local_rand_state);
        }

        __device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {
        }

        // Class "getters"
        __device__
        vec3 origin() const {
            return orig;
        }
        __device__
        vec3 direction() const {
            return dir;
        }

        // Class methods
        __device__
        vec3 at(float t) const {
            /* returns the coordinates of the point at distance t from the ray origin */
            return orig + t*dir;
        };

        __device__
        void init_spectrum(curandState* local_rand_state) {
            init_hero_wavelength(wavelengths, N_RAY_WAVELENGTHS, local_rand_state);
            for (float & i : power_distr) {
                i = 1.0f;
            }
            wavelengths_zeroed = false;
        }

        __device__
        void non_hero_to_zero() {
            for (int i = 1; i < N_RAY_WAVELENGTHS; i++) {
                power_distr[i] = 0.f;
            }
            wavelengths_zeroed = true;
        }

        __device__
        void mul_spectrum(const float* spectral_distr) {
            if (!wavelengths_zeroed) {
                float lambda;
                float weight;
                for (int i = 0; i < N_RAY_WAVELENGTHS; i++) {
                    lambda = wavelengths[i];
                    weight = spectrum_interp(spectral_distr, lambda);
                    power_distr[i] *= weight;
                }
            }
            else {
                power_distr[0] *= spectrum_interp(spectral_distr, wavelengths[0]);
            }
        }

        __device__
        void mul_spectrum(float value) {
            if (!wavelengths_zeroed) {
                for (int i = 0; i < N_RAY_WAVELENGTHS; i++) {
                    power_distr[i] *= value;
                }
            } else {
                power_distr[0] *= value;
            }
        }
    };


#endif //SPECTRAL_RT_PROJECT_RAY_CUH

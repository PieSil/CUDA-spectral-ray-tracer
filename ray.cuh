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
        }
    };


#endif //SPECTRAL_RT_PROJECT_RAY_CUH

//
// Created by pietr on 08/04/2024.
//

#ifndef SPECTRAL_RT_PROJECT_RAY_CUH
#define SPECTRAL_RT_PROJECT_RAY_CUH

#include "vec3.cuh"
#include "spectrum.cuh"
#include "cie_const.cuh"

#define N_RAY_WAVELENGTHS 7

class ray {
public:
	vec3 orig;
	vec3 dir;
	uint valid_wavelengths = N_RAY_WAVELENGTHS;
	float wavelengths[N_RAY_WAVELENGTHS] = { 0.0f };
	float power_distr[N_RAY_WAVELENGTHS] = { 0.0f };

	__device__ ray() {};

	__device__
		ray(const point3& origin, const vec3& direction, curandState* local_rand_state) : orig(origin), dir(direction) {
		init_spectrum(local_rand_state);
	}

	__device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {
	}

	__device__
		vec3 origin() const {
		return orig;
	}
	__device__
		vec3 direction() const {
		return dir;
	}

	__device__
		vec3 at(float t) const {
		/* returns the coordinates of the point at distance t from the ray origin */
		return orig + t * dir;
	};

	__device__
		void init_spectrum(curandState* local_rand_state) {
		init_hero_wavelength(wavelengths, N_RAY_WAVELENGTHS, local_rand_state);
		for (float& i : power_distr) {
			i = 1.0f;
		}
		valid_wavelengths = N_RAY_WAVELENGTHS;
	}

	__device__
		void mul_spectrum(const float* spectral_distr, const uint n_samples) {
		float lambda;
		float weight;
		for (int i = 0; i < valid_wavelengths; i++) {
			lambda = wavelengths[i];
			weight = spectrum_interp(spectral_distr, lambda, n_samples);
			power_distr[i] *= weight;
		}
	}

	__device__
		void mul_spectrum(float value) {
			for (int i = 0; i < valid_wavelengths; i++) {
				power_distr[i] *= value;
			}
	
	}
};


#endif //SPECTRAL_RT_PROJECT_RAY_CUH

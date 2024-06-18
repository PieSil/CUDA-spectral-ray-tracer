//
// Created by pietr on 19/03/2024.
//

#ifndef RTWEEKEND_CUDA_MATERIAL_CUH
#define RTWEEKEND_CUDA_MATERIAL_CUH


#include "color.cuh"
#include "hit_record.cuh"
#include "color_to_spectrum.cuh"
#include "sellmeier.cuh"

#define EPSILON 0.0001f

#define LAMBERTIAN 0
#define METALLIC 1
#define DIELECTRIC 2
//#define DIELECTRIC_CONST 3
#define EMISSIVE 4
//#define NORMAL_TEST 5
#define NO_MAT 6

class material {
public:
    __host__ __device__
    material() : col(0.0f, 0.0f, 0.0f), reflection_fuzz(1.0f),
                 material_type(NO_MAT), emission_power(0.0f) {
        for (int i = 0; i < 3; i++) {
            sellmeier_B[i] = 0.f;
            sellmeier_C[i] = 0.f;
        }
    }

    __device__
        material(color _col, float fuzz, float ir, float power, uint type) : col(_col), reflection_fuzz(fuzz),
        material_type(type), emission_power(power) {
        for (int i = 0; i < 3; i++) {

            if (i == 0)
                sellmeier_B[i] = ir;
            else 
                sellmeier_B[i] = 0.f;

            sellmeier_C[i] = 0.f;
        }

    }

    __device__
    material(color _col, float fuzz, const float b[3], const float c[3], float power, uint type) : col(_col), reflection_fuzz(fuzz),
                 material_type(type), emission_power(power) {
        for (int i = 0; i < 3; i++) {
            sellmeier_B[i] = b[i];
            sellmeier_C[i] = b[i];
        }
    }

    __device__
    void compute_spectral_distr(float* dev_sRGBToSpectrum_Data) {
        switch (material_type) {
        case EMISSIVE:
            dev_srgb_to_illuminance_spectrum(col, spectral_distribution, emission_power, dev_sRGBToSpectrum_Data);
            break;
        case DIELECTRIC:
            setSpectralDistribution(1.0f);
            break;
        default:
            dev_srgb_to_spectrum(col, spectral_distribution, dev_sRGBToSpectrum_Data);
            break;
        }
    }

    __device__
    void setSpectralDistribution(const float* spectrum) {
        for(int i = 0; i < N_CIE_SAMPLES; i++) {
            spectral_distribution[i] = spectrum[i];
        }
    }

    __device__
        void setSpectralDistribution(const float value) {
        for (int i = 0; i < N_CIE_SAMPLES; i++) {
            spectral_distribution[i] = value;
        }
    }

    __device__
    static material emissive(const color emitted, const float power = 1.0f) {
        return material(emitted, 1.0f, 1.0f, power, EMISSIVE);
    }

    __device__
    static material lambertian(const color col) {
        return material(col,  1.0f, 1.0f, 0.0f, LAMBERTIAN);
    }

    __device__
    static material metallic(const color col, const float fuzz) {
        return material(col, fuzz, 1.0f, 0.0f, METALLIC);
    }
    __device__
    static material dielectric(const float b[3], const float c[3]) {
        return material(color(1.0f, 1.0, 1.0f), 1.0f, b, c, 0.0f, DIELECTRIC);
    }

    __device__
    const bool scatter(ray &r_in, const hit_record &rec, curandState *local_rand_state) const;

    __device__
    bool unified_scatter(ray& r_in, const hit_record& rec, curandState* local_rand_state) const;

    color col;
    float reflection_fuzz;
    uint material_type;
    float spectral_distribution[N_CIE_SAMPLES];
    float emission_power;

    //Sellemeier's equation coefficients
    float sellmeier_B[3]; //value at index 0 is reused as the refractive index in case of DIELECTRIC_CONST material
    float sellmeier_C[3];
    
};



__device__
const bool refraction_scatter(float mat_ir, const hit_record &rec, float &epsilon_correction_sign, vec3 &scatter_direction, vec3 unit_in_direction,
                   curandState *local_rand_state);

__device__
const float reflectance(float cosine, float ref_idx);

__device__
const bool reflection_scatter(float mat_fuzz, vec3 unit_in_direction, const hit_record &rec, vec3 &scattered_direction,
                              curandState *local_rand_state);

__device__
const bool
lambertian_scatter(const hit_record &rec, vec3 &scatter_direction, curandState *local_rand_state);


#endif //RTWEEKEND_CUDA_MATERIAL_CUH

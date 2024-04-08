//
// Created by pietr on 19/03/2024.
//

#ifndef RTWEEKEND_CUDA_MATERIAL_CUH
#define RTWEEKEND_CUDA_MATERIAL_CUH


#include "color.cuh"
#include "hittable.cuh"
#include "color_to_spectrum.cuh"

#define EPSILON 0.0001f

enum MAT_TYPE {
    LAMBERTIAN,
    METALLIC,
    DIELECTRIC,
    GENERIC,
    EMISSIVE
};

class material {
public:
    __host__ __device__
    material() : albedo(1.0f, 1.0f, 1.0f), reflection_fuzz(1.0f), emitted(0.0f, 0.0f, 0.0f),
                 material_type(MAT_TYPE::LAMBERTIAN), ir(1.0f),emission_power(1.0f) {
    }

    __device__
    material(color _albedo, float fuzz, color emit, float _ir, float power, MAT_TYPE type) : albedo(_albedo), reflection_fuzz(fuzz), emitted(emit),
                 material_type(type), ir(_ir), emission_power(power) {
    }

    __device__
    void compute_albedo_spectrum(float* dev_sRGBToSpectrum_Data) {
        dev_srgb_to_spectrum(albedo, spectral_reflectance_distribution, dev_sRGBToSpectrum_Data);
//        printf("albedo: (%f, %f, %f)\nReflectance distribution:\n", albedo[0], albedo[1], albedo[2]);
//        float lambda = LAMBDA_MIN;
//        for(int i = 0; i < N_CIE_SAMPLES; i++) {
//            printf("lambda: %f, value: %f\n", lambda, spectral_reflectance_distribution[i]);
//            lambda += 5.0f;
//        }
//        printf("\n");
    }

    __device__
    void compute_emittance_spectrum(float* dev_sRGBToSpectrum_Data) {
        dev_srgb_to_illuminance_spectrum(emitted, spectral_emittance_distribution, emission_power, dev_sRGBToSpectrum_Data);
    }

    __device__
    void setEmittanceSpectrum(const float* spectrum) {
        for(int i = 0; i < N_CIE_SAMPLES; i++) {
            spectral_emittance_distribution[i] = spectrum[i];
        }
    }

    __device__
    void setReflectanceSpectrum(const float* spectrum) {
        for(int i = 0; i < N_CIE_SAMPLES; i++) {
            spectral_reflectance_distribution[i] = spectrum[i];
        }
    }

    __device__
    static material emissive(const color emitted, const float power = 1.0f) {
        return material(color(1.0f, 1.0f, 1.0f),  1.0f, emitted, 1.0f, power, EMISSIVE);
    }

    __device__
    static material lambertian(const color albedo) {
        return material(albedo,  1.0f, color(0.0f, 0.0f, 0.0f), 1.0f, 0.0f, LAMBERTIAN);
    }

    __device__
    static material metallic(const color albedo, const float fuzz) {
        return material(albedo,  fuzz, color(0.0f, 0.0f, 0.0f), 1.0f, 0.0f, METALLIC);
    }
    __device__
    static material dielectric(const float ir) {
        return material(color(1.0f, 1.0, 1.0f), 1.0f, color(0.0f, 0.0f, 0.0f), ir, 0.0f, DIELECTRIC);
    }

    __host__
    const void to_device(material* dev_ptr) {
        checkCudaErrors(cudaMemcpy(dev_ptr, this, sizeof(material), cudaMemcpyHostToDevice));
    }

    __host__
    const void to_host(material* host_ptr) {
        checkCudaErrors(cudaMemcpy(host_ptr, this, sizeof(material), cudaMemcpyDeviceToHost));
    }

    __device__
    const bool scatter(ray &r_in, const hit_record &rec, curandState *local_rand_state) const;

    //TODO: maybe find a way to move data into shared memory?
    color albedo;
    color emitted;
    float reflection_fuzz;
    MAT_TYPE material_type;
    float ir;
    float spectral_reflectance_distribution[N_CIE_SAMPLES];
    float spectral_emittance_distribution[N_CIE_SAMPLES];
    float emission_power;
};



__device__
const bool refraction_scatter(float mat_ir, const hit_record &rec, point3 &scatter_origin, vec3 &scatter_direction,
                   vec3 unit_in_direction, curandState *local_rand_state);

__device__
const float reflectance(float cosine, float ref_idx);

__device__
const bool reflection_scatter(float mat_fuzz, vec3 unit_in_direction, const hit_record &rec, vec3 &scattered_direction,
                              curandState *local_rand_state);

__device__
const bool
lambertian_scatter(const hit_record &rec, vec3 &scatter_direction, curandState *local_rand_state);


#endif //RTWEEKEND_CUDA_MATERIAL_CUH

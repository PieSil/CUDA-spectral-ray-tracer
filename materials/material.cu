//
// Created by pietr on 19/03/2024.
//

#include "material.cuh"

//lambertian
__device__
const bool
lambertian_scatter(const hit_record& rec, vec3& scatter_direction, curandState* local_rand_state) {
    scatter_direction = rec.normal + random_unit_vector(local_rand_state);

    /*Catch degenerate scatter direction (normal + unit vector too close to 0) */
    if (scatter_direction.near_zero())
        scatter_direction = rec.normal;

    //attenuation = (attenuation * weight);
    return true;
}

//reflection
__device__
const bool reflection_scatter(float mat_fuzz, vec3 unit_in_direction, const hit_record& rec, vec3& scattered_direction,
    curandState* local_rand_state) {
    vec3 reflected = reflect(unit_in_direction, rec.normal);

    /*
     * Sum a random vector centered on the reflected vector's endpoint scaled by the fuzz factor
     */
    scattered_direction = reflected + mat_fuzz * random_unit_vector(local_rand_state);

    /*
    * For big fuzz values the scattered ray may go below the surface,
    * if this happens just let the surface absorb the ray
    */
    return (dot(scattered_direction, rec.normal) > 0);
}

__device__
const float reflectance(float cosine, float ref_idx) {
    //Use Schlick's approximation for reflectance
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx); //reflectance of light when the incident ray
    // is perpendicular to the surface (i.e. cosine = 0)

    r0 = r0 * r0; //part of Schlick's approximation, adjusts r0 in order to account
    // for the behaviour of light at oblique angles

    return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f); //r0: reflectance at normal incidence
    // (1 - r0): reduction in reflectance as the angle of
    //          incidence deviates from normal
    // pow(1- cosine, 5): models the decrease in reflectance
    //                    as the angle of incidence increases
}

__device__
const bool material::scatter(ray& r_in, const hit_record& rec, curandState* local_rand_state) const {
    vec3 scatter_direction;
    float epsilon_correction_sign = 1.0f;
    bool did_scatter = true;
    vec3 unit_in_direction = unit_vector(r_in.direction());

    switch (material_type) {

        case METALLIC:
             
            did_scatter = reflection_scatter(reflection_fuzz, unit_in_direction, rec, scatter_direction,
                                            local_rand_state);

            if (!did_scatter)
                r_in.valid_wavelengths = 0;
        break;
        
        case DIELECTRIC:
            float ir = sellmeier_index(sellmeier_B, sellmeier_C, r_in.wavelengths[0]);

            bool refracted = refraction_scatter(ir, rec, epsilon_correction_sign, scatter_direction, unit_in_direction, local_rand_state);

            if (refracted)
                r_in.valid_wavelengths = 1;
            break;
       

        case EMISSIVE:

            did_scatter = false;
            break;

        case LAMBERTIAN:
        default:

            lambertian_scatter(rec, scatter_direction, local_rand_state);
            break;
    }

    r_in.mul_spectrum(spectral_distribution, N_CIE_SAMPLES);
    r_in.orig = rec.p + epsilon_correction_sign * EPSILON * rec.normal;
    r_in.dir = scatter_direction;

    return did_scatter;
}

    __device__
    const bool
    refraction_scatter(const float mat_ir, const hit_record &rec, float &epsilon_correction_sign, vec3 &scatter_direction, const vec3 unit_in_direction,
                       curandState *local_rand_state)

    {
        float refraction_ratio = rec.front_face ? (1.0f/mat_ir) : mat_ir;

        //check if Snell's law has solution or not (refraction_ratio * sin_theta must be <= 1)
        float cos_theta = fmin(dot(-unit_in_direction, rec.normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f || reflectance(cos_theta, refraction_ratio) > cuda_random_float(local_rand_state);



        /*
         * We reflect the ray if Snell's law has no solution or if the reflectance index (computed by reflectance()) is greater than a random float
         * since cuda_random_float() generates values in [0.0, 1.0) with a uniform distribution we have that the greater
         * the reflectance index the more likely it will be to refract instead of reflect
         */
        if (cannot_refract) {
            scatter_direction = reflect(unit_in_direction, rec.normal);

            // scatter_origin = rec.p + EPSILON * rec.normal;
        } else {
            scatter_direction = refract(unit_in_direction, rec.normal, refraction_ratio);

            epsilon_correction_sign = -1.0f;
        }



        return !cannot_refract;
    }

    __device__
        bool material::unified_scatter(ray& r_in, const hit_record& rec, curandState* local_rand_state) const {
        vec3 lambertian_scatter_direction;
        vec3 metallic_scatter_direction;
        vec3 dielectric_scatter_direction;
        float epsilon_correction_sign = 1.0f;
        bool did_scatter;
        vec3 unit_in_direction = unit_vector(r_in.direction());
        
        did_scatter = reflection_scatter(reflection_fuzz, unit_in_direction, rec, metallic_scatter_direction, local_rand_state);
        bool refracted = refraction_scatter(sellmeier_index(sellmeier_B, sellmeier_C, r_in.wavelengths[0]), rec, epsilon_correction_sign, dielectric_scatter_direction, unit_in_direction, local_rand_state);
        lambertian_scatter(rec, lambertian_scatter_direction, local_rand_state);

        float lambertian_weight = 0.0f;
        float metallic_weight = 0.0f;
        float dielectric_weight = 0.0f;

        switch (material_type) {
        case LAMBERTIAN:
            lambertian_weight = 1.0f;
            did_scatter = true;
            break;
        case METALLIC:
            metallic_weight = 1.0f;
            if (!did_scatter)
                r_in.valid_wavelengths = 0;
            break;
        case DIELECTRIC:
            dielectric_weight = 1.0f;
            did_scatter = true;
            if (refracted)
                r_in.valid_wavelengths = 1;
            break;
        case EMISSIVE:
            did_scatter = false;
        default:
        }

        //if material is dielectric apply computed correction sign, otherwise correction sign is always 1
        r_in.orig = rec.p + (dielectric_weight * epsilon_correction_sign + (1.f-dielectric_weight)) * EPSILON * rec.normal;
        //choose scatter direction based on weight
        r_in.dir = lambertian_weight * lambertian_scatter_direction + metallic_weight * metallic_scatter_direction + dielectric_weight * dielectric_scatter_direction;
        r_in.mul_spectrum(spectral_distribution, N_CIE_SAMPLES);

        return did_scatter;
    }
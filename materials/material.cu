//
// Created by pietr on 19/03/2024.
//

#include "material.cuh"

//lambertian
__device__
const bool
lambertian_scatter(const hit_record &rec, vec3 &scatter_direction, curandState *local_rand_state) {
    scatter_direction = rec.normal + random_unit_vector(local_rand_state);

    /*Catch degenerate scatter direction (normal + unit vector too close to 0) */
    if (scatter_direction.near_zero())
        scatter_direction = rec.normal;

    //attenuation = (attenuation * weight);
    return true;
}

//reflection
__device__
const bool reflection_scatter(float mat_fuzz, vec3 unit_in_direction, const hit_record &rec, vec3 &scattered_direction,
                              curandState *local_rand_state) {
    vec3 reflected = reflect(unit_in_direction, rec.normal);

    /*
     * Sum a random vector centered on the reflected vector's endpoint scaled by the fuzz factor
     */
    scattered_direction = reflected + mat_fuzz * random_unit_vector(local_rand_state);
    //attenuation = (attenuation * weight);

    /*
    * For big fuzz values the scattered ray may go below the surface,
    * if this happens just let the surface absorb the ray
    */
    return (dot(scattered_direction, rec.normal ) > 0);
}

__device__
const float reflectance(float cosine, float ref_idx) {
    //Use Schlick's approximation for reflectance
    auto r0 = (1.0f-ref_idx) / (1.0f+ref_idx); //reflectance of light when the incident ray
    // is perpendicular to the surface (i.e. cosine = 0)

    r0 = r0*r0; //part of Schlick's approximation, adjusts r0 in order to account
    // for the behaviour of light at oblique angles

    return r0 + (1.0f-r0)*pow(1.0f-cosine, 5.0f); //r0: reflectance at normal incidence
    // (1 - r0): reduction in reflectance as the angle of
    //          incidence deviates from normal
    // pow(1- cosine, 5): models the decrease in reflectance
    //                    as the angle of incidence increases
}

__device__
const bool material::scatter(ray &r_in, const hit_record &rec, curandState *local_rand_state) const {
    vec3 scatter_direction;
    point3 scatter_origin = rec.p + EPSILON * rec.normal;
//    vec3 reflection_direction;
//    vec3 diffuse_direction;
    bool did_scatter;
    vec3 unit_in_direction = unit_vector(r_in.direction());
    //float random = cuda_random_float(local_rand_state);
    float lambda;
    float weight;

    //TODO: UNIFY AS MUCH AS POSSIBLE
    switch(material_type) {

        case MAT_TYPE::METALLIC:
            //TODO: check if total light reflection needs to be performed
            
             
            did_scatter = reflection_scatter(reflection_fuzz, unit_in_direction, rec, scatter_direction,
                                            local_rand_state);

            for (int i = 0; i < N_RAY_WAVELENGTHS; i++) {
                lambda = r_in.wavelengths[i];
                weight = did_scatter ? spectrum_interp(spectral_reflectance_distribution, lambda) : 0.0f;
                r_in.power_distr[i] *= weight;
            }
        break;

        case MAT_TYPE::DIELECTRIC:
            did_scatter = refraction_scatter(ir, rec, scatter_origin, scatter_direction, unit_in_direction,
                                             local_rand_state);
        break;

        case MAT_TYPE::EMISSIVE:
            /*
            weight = spectrum_interp(spectral_emittance_distribution, r_in.wavelength);

            if (random > weight)
                new_wl = 0.0f;
            */

            did_scatter = false;
            break;

        case MAT_TYPE::LAMBERTIAN:
        default:

            for (int i = 0; i < N_RAY_WAVELENGTHS; i++) {
                lambda = r_in.wavelengths[i];
                weight = spectrum_interp(spectral_reflectance_distribution, lambda);
                r_in.power_distr[i] *= weight;
            }

//            for (int i = 0; i < N_RAY_WAVELENGTHS; i++) {
//                r_in.wl_pdf[i] /= sum_pdf;
//            }

            did_scatter = lambertian_scatter(rec, scatter_direction, local_rand_state);
            break;
    }

//    if (try_refract(mat.ir, rec, unit_in_direction, refraction_ratio, local_rand_state, transmittance)){
//        refraction_scatter(rec, final_direction, unit_in_direction, vec3(), nullptr);
//        did_scatter = (dot(scattered.direction(), rec.normal ) > 0);
//    } else {
//        lambertian_scatter(rec, scatter_direction, attenuation, local_rand_state);
//        reflection_scatter(r_in, rec, reflect_direction, attenuation);
//
//        //interpolate directions based on shininess factor
//        final_direction = mat.reflection_fuzz * reflect_direction + (1 - mat.reflection_fuzz) * scatter_direction;
//
//        /*
//         * For big fuzz values the scattered ray may go below the surface,
//         * if this happens just let the surface absorb the ray
//         */


    r_in.orig = scatter_origin;
    r_in.dir = scatter_direction;

    return did_scatter;
}

    __device__
    const bool
    refraction_scatter(const float mat_ir, const hit_record &rec, point3 &scatter_origin, vec3 &scatter_direction,
                       const vec3 unit_in_direction, curandState *local_rand_state)

    {
        //attenuation = attenuation * color(1.0f, 1.0f, 1.0f); //no attenuation
        float refraction_ratio = rec.front_face ? (1.0f/mat_ir) : mat_ir;

        //check if Snell's law has solution or not (refraction_ratio * sin_theta must be <= 1)
        float cos_theta = fmin(dot(-unit_in_direction, rec.normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;


        /*
         * We reflect the ray if Snell's law has no solution or if the reflectance index is greater than a random float
         * since cuda_random_float() generates values in [0.0, 1.0) with a uniform distribution we have that the greater
         * the reflectance index the more likely it will be to refract instead of reflect
         */


        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > cuda_random_float(local_rand_state)) {
            scatter_direction = reflect(unit_in_direction, rec.normal);
            // scatter_origin = rec.p + EPSILON * rec.normal;
        } else {
            scatter_direction = refract(unit_in_direction, rec.normal, refraction_ratio);
            //scatter_origin = rec.p - EPSILON * rec.normal;
        }

        scatter_origin = rec.p + EPSILON * scatter_direction;

        return true;
    }
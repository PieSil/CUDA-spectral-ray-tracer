//
// Created by pietr on 20/05/2024.
//

#ifndef SPECTRAL_RT_PROJECT_HIT_RECORD_CUH
#define SPECTRAL_RT_PROJECT_HIT_RECORD_CUH

#include "ray.cuh"
#include "aabb.cuh"

struct material;

class hit_record {
public:

    hit_record() = default;
    hit_record(const hit_record& other) = default;
    hit_record& operator=(const hit_record& r) = default;

    point3 p;
    vec3 normal;
    float t;
    //float u, v; //surface coordinates of the ray-object hit point
    bool front_face;

    material* mat; //When a ray hits a surface the material pointer will be set to point
    // at the material the surface was given


    __device__
    void set_face_normal(const ray& r, const vec3& outward_normal) {
        /*
         * Sets the hit record normal vector.
         * NOTE: the parameter 'outward_normal' is assumed to have unit length.
         *
         * Compute the dot product between the ray and the normal facing outward
         * if it's less than 0 then the ray is outside the object,
         * otherwise the ray is inside the object and we need to reverse the normal direction
         * (as we want normals to always face towards the ray origin)
         */

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

#endif //SPECTRAL_RT_PROJECT_HIT_RECORD_CUH

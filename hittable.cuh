//
// Created by pietr on 08/04/2024.
//

#ifndef SPECTRAL_RT_PROJECT_HITTABLE_CUH
#define SPECTRAL_RT_PROJECT_HITTABLE_CUH

#include "ray.cuh"

class hit_record {
public:
    point3 p;
    vec3 normal;
    float t;
    //float u, v; //surface coordinates of the ray-object hit point
    bool front_face;
    // material* mat; //When a ray hits a surface the material pointer will be set to point
    // at the material the surface was given
    //TODO: remove pointers if possible in order to avoid access to global memory (how: transform the classes into structs and create "standalone" functions)


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

class hittable {

};


#endif //SPECTRAL_RT_PROJECT_HITTABLE_CUH

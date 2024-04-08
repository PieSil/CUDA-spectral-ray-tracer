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
    /* A generic class for things that can be hit by a ray */

    __host__ __device__
    virtual ~hittable() {} // DO NOT USE = default, it breaks everything, I lack the knowledge and the will
    // to understand why (since it should be the same thing), but it appears it isn't.

    __device__
    virtual bool hit(const ray& r, float min, float max, hit_record& rec) const = 0;

    // __host__ __device__
    // virtual aabb bounding_box() const = 0;
};


#endif //SPECTRAL_RT_PROJECT_HITTABLE_CUH

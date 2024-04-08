//
// Created by pietr on 08/04/2024.
//

#ifndef SPECTRAL_RT_PROJECT_SPHERE_CUH
#define SPECTRAL_RT_PROJECT_SPHERE_CUH

#include "hittable.cuh"
#include "vec3.cuh"
#include "interval.cuh"

class sphere : public hittable {
public:
    __host__ __device__
    sphere(point3 _center, float _radius, material* _material) : center(_center), radius(_radius), mat(_material) {
        //TODO: if adding motion blur revise this
        auto rvec = vec3(radius, radius, radius);
        bbox = aabb(center - rvec, center + rvec);
    }

    __device__
    bool hit(const ray &r, float min, float max, hit_record &rec) const override;

    __host__ __device__
    aabb bounding_box() const override {
        return bbox;
    }

private:
    point3 center;
    float radius;
    material* mat;
    aabb bbox;
};


#endif //SPECTRAL_RT_PROJECT_SPHERE_CUH

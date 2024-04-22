//
// Created by pietr on 28/12/2023.
//

#ifndef RTWEEKEND_CUDA_QUAD_CUH
#define RTWEEKEND_CUDA_QUAD_CUH

#include "hittable.cuh"
#include "materials/material.cuh"

class quad : public hittable {
public:

    __device__
    quad(const point3& _Q, const vec3& _u, const vec3& _v, material* m) : Q(_Q), u(_u), v(_v), mat(m) {
        init();
    }

    __host__ __device__
    void init() {
        //plane implicit formula is Ax + By + Cz = D where:
        //(A, B, C) is the plane normal
        //D is a constant
        //(x, y, z) are the coordinates of a point on the plane
        //thus we need to compute the plane normal and find D

        auto n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q); //solves D = Q_x*n_x + Q_y*n_y + Q_z*n_z in order to find D
        w = n / dot(n, n);

        set_bounding_box();
    }


    __host__ __device__
    virtual void set_bounding_box() {
        bbox = aabb(Q, Q+u+v).pad();
    }

    __host__ __device__
    aabb bounding_box() const override {
        return bbox;
    }

    __device__
    bool hit(const ray& r, float min, float max, hit_record& rec) const override;

    __device__
    virtual bool is_interior(double a, double b, hit_record& rec) const {
        // Given the hit point in plane coordinates, return false if it is outside the
        // primitive, otherwise set the hit record UV coordinates and return true.

        if ((a < 0) || (1 < a) || (b < 0) || (1 < b))
            return false;

        // rec.u = a; //needed for texture support, not introduced yet
        // rec.v = b; //needed for texture support, not introduced yet
        return true;
    }

    __host__ __device__
    point3 center() {

        return ((u + v) / 2.0f) + Q;;
    }

    point3 Q; //origin point Q
    vec3 u, v; //vectors with origin in Q that define the "bottom left" corner of the quad
    material* mat;
    aabb bbox;
    vec3 normal;
    float D;
    vec3 w; //cached value needed during ray intersection computation
};


#endif //RTWEEKEND_CUDA_QUAD_CUH

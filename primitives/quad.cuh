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
    virtual bool is_interior(float a, float b) const {
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

        return ((u + v) / 2.0f) + Q;
    }

    point3 Q; //origin point Q
    vec3 u, v; //vectors with origin in Q that define the "bottom left" corner of the quad
    material* mat;
    aabb bbox;
    vec3 normal;
    float D;
    vec3 w; //cached value needed during ray intersection computation
};

struct box {
    __device__
    box(const point3& a, const point3& b, material* mat, quad** _sides) {
        point3 min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
        point3 max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

        vec3 dx = vec3(max.x() - min.x(), 0.f, 0.f);
        vec3 dy = vec3(0, max.y() - min.y(), 0.f);
        vec3 dz = vec3(0, 0, max.z() - min.z());

        _sides[0] = new quad(point3(min.x(), min.y(), max.z()), dx, dy, mat); // front
        _sides[1] = new quad(point3(max.x(), min.y(), max.z()), -dz, dy, mat); // right
        _sides[2] = new quad(point3(max.x(), min.y(), min.z()), -dx, dy, mat); // back
        _sides[3] = new quad(point3(min.x(), min.y(), min.z()), dz, dy, mat); // left
        _sides[4] = new quad(point3(min.x(), max.y(), max.z()), dx, -dz, mat); // top
        _sides[5] = new quad(point3(min.x(), min.y(), min.z()), dx, dz, mat); // bottom
        sides = _sides;
    }

    __device__
    box(quad** _sides) {
        sides = _sides;
    }

    __device__
    box(const point3 center, vec3 width_vec, vec3 height_vec, vec3 depth_vec, material* mat, quad** _sides) {
        vec3 half_sum = (width_vec + height_vec + depth_vec) / 2.0f;
        point3 min = center - half_sum;

        //point3 min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
        //point3 max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

        _sides[0] = new quad(point3(min.x(), min.y(), min.z()) + depth_vec, width_vec, height_vec, mat); // front
        _sides[1] = new quad(point3(min.x(), min.y(), min.z()) + depth_vec + width_vec, -depth_vec, height_vec, mat); // right
        _sides[2] = new quad(point3(min.x(), min.y(), min.z()) + width_vec, -width_vec, height_vec, mat); // back
        _sides[3] = new quad(point3(min.x(), min.y(), min.z()), depth_vec, height_vec, mat); // left
        _sides[4] = new quad(point3(min.x(), min.y(), min.z()) + height_vec + depth_vec, width_vec, -depth_vec, mat);// top
        _sides[5] = new quad(point3(min.x(), min.y(), min.z()), width_vec, depth_vec, mat); // bottom

        sides = _sides;
    }

    __device__
    box(const box& outer, material* mat, quad** _sides, const float offset = 0.0f) {
        
        vec3 outer_width_vec = outer.getWidthVec();
        vec3 outer_height_vec = outer.getHeightVec();
        vec3 outer_depth_vec = outer.getDepthVec();


        box(outer.getCenter(), outer_width_vec - offset * unit_vector(outer_width_vec), outer_height_vec - offset * unit_vector(outer_height_vec), outer_depth_vec - offset * unit_vector(outer_depth_vec), mat, _sides);
        sides = _sides;
    }

    __host__ __device__
    const point3 getLocalMin() const {
        return sides[5]->Q;
    }

    __host__ __device__
    const point3 getLocalMax() const {
        point3 min = getLocalMin();
        return min + getWidthVec() + getHeightVec() + getDepthVec();
        //return point3(sides[4]->Q.e + sides[4]->u, sides[4]->Q.e[1], sides[4]->Q.e[2]);
    }

    __host__ __device__
        const point3 getLocalMax(const point3 min) const {
        return min + getWidthVec() + getHeightVec() + getDepthVec();
        //return point3(sides[4]->Q.e + sides[4]->u, sides[4]->Q.e[1], sides[4]->Q.e[2]);
    }

    __host__ __device__
    const point3 getCenter() const {
        point3 min = getLocalMin();
        point3 max = getLocalMax(min);
        point3 center = (max - min) / 2.0f + min;
        return center;
    }

    __host__ __device__
    const vec3 getWidthVec() const {
        return sides[5]->u;
    }

    __host__ __device__
    const vec3 getHeightVec() const {
        return sides[3]->v;
    }

    __host__ __device__
    const vec3 getDepthVec() const {
        return sides[5]->v;
    }

    quad** sides;
};


#endif //RTWEEKEND_CUDA_QUAD_CUH

#ifndef SPECTRAL_RT_PROJECT_TRI_BOX_CUH
#define SPECTRAL_RT_PROJECT_TRI_BOX_CUH

#include "tri_quad.cuh"

class tri_box {

public:

    __device__
    tri_box(const point3& a, const point3& b, material* mat, tri** tris, const bool defer_init = false) {
        point3 min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
        point3 max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

        vec3 dx = vec3(max.x() - min.x(), 0.f, 0.f);
        vec3 dy = vec3(0, max.y() - min.y(), 0.f);
        vec3 dz = vec3(0, 0, max.z() - min.z());

        //const point3& Q, const vec3& u, const vec3& v, material* m, tri** _halves, bool defer_init = false
        sides[0] = tri_quad(point3(min.x(), min.y(), max.z()), dx, dy, mat, &tris[0], defer_init); // front
        sides[1] = tri_quad(point3(max.x(), min.y(), max.z()), -dz, dy, mat, &tris[2], defer_init); // right
        sides[2] = tri_quad(point3(max.x(), min.y(), min.z()), -dx, dy, mat, &tris[4], defer_init); // back
        sides[3] = tri_quad(point3(min.x(), min.y(), min.z()), dz, dy, mat, &tris[6], defer_init); // left
        sides[4] = tri_quad(point3(min.x(), max.y(), max.z()), dx, -dz, mat, &tris[8], defer_init); // top
        sides[5] = tri_quad(point3(min.x(), min.y(), min.z()), dx, dz, mat, &tris[10], defer_init); // bottom
    }

    __device__
    tri_box(const point3 center, vec3 width_vec, vec3 height_vec, vec3 depth_vec, material* mat, tri** tris, const bool defer_init = false) {
        vec3 half_sum = (width_vec + height_vec + depth_vec) / 2.0f;
        point3 min = center - half_sum;

        //point3 min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
        //point3 max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

        sides[0] = tri_quad(point3(min.x(), min.y(), min.z()) + depth_vec, width_vec, height_vec, mat, &tris[0], defer_init); // front
        sides[1] = tri_quad(point3(min.x(), min.y(), min.z()) + depth_vec + width_vec, -depth_vec, height_vec, mat, &tris[2], defer_init); // right
        sides[2] = tri_quad(point3(min.x(), min.y(), min.z()) + width_vec, -width_vec, height_vec, mat, &tris[4], defer_init); // back
        sides[3] = tri_quad(point3(min.x(), min.y(), min.z()), depth_vec, height_vec, mat, &tris[6], defer_init); // left
        sides[4] = tri_quad(point3(min.x(), min.y(), min.z()) + height_vec + depth_vec, width_vec, -depth_vec, mat, &tris[8], defer_init);// top
        sides[5] = tri_quad(point3(min.x(), min.y(), min.z()), width_vec, depth_vec, mat, &tris[10], defer_init); // bottom

    }

    __device__
    tri_box(tri_box outer, material* mat, tri** tris, const float offset = 0.0f, const bool defer_init = false) {

        vec3 outer_width_vec = outer.getWidthVec();
        vec3 outer_height_vec = outer.getHeightVec();
        vec3 outer_depth_vec = outer.getDepthVec();

        *this = tri_box(outer.center(), outer_width_vec - offset * unit_vector(outer_width_vec), outer_height_vec - offset * unit_vector(outer_height_vec), outer_depth_vec - offset * unit_vector(outer_depth_vec), mat, tris, defer_init);
    }

    __device__
    tri_box init() {
        sides[0].init();
        sides[1].init();
        sides[2].init();
        sides[3].init();
        sides[4].init();
        sides[5].init();
    }

    __device__
        const point3 getLocalMin() const {
        return sides[5].Q();
    }

    __device__
        const point3 getLocalMax() const {
        point3 min = getLocalMin();
        return min + getWidthVec() + getHeightVec() + getDepthVec();
        //return point3(sides[4]->Q.e + sides[4]->u, sides[4]->Q.e[1], sides[4]->Q.e[2]);
    }

    __device__
        const point3 getLocalMax(const point3 min) const {
        return min + getWidthVec() + getHeightVec() + getDepthVec();
        //return point3(sides[4]->Q.e + sides[4]->u, sides[4]->Q.e[1], sides[4]->Q.e[2]);
    }

    __device__
        const point3 center() const {
        point3 min = getLocalMin();
        point3 max = getLocalMax(min);
        point3 center = (max - min) / 2.0f + min;
        return center;
    }

    __device__
        const vec3 getWidthVec() const {
        return sides[5].u();
    }

    __device__
        const vec3 getHeightVec() const {
        return sides[3].v();
    }

    __device__
        const vec3 getDepthVec() const {
        return sides[5].v();
    }

    __device__
    void flip_normals() {
        sides[0].flip_normals();
        sides[1].flip_normals();
        sides[2].flip_normals();
        sides[3].flip_normals();
        sides[4].flip_normals();
        sides[5].flip_normals();
    }

    __device__
    void translate(const vec3 dir, const bool reinit = true);

    __device__
    void rotate(const float theta, const transform::AXIS ax, const bool reinit = true, const bool local= true);

    tri_quad sides[6];
};

#endif
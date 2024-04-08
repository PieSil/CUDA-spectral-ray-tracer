//
// Created by pietr on 08/04/2024.
//

#ifndef SPECTRAL_RT_PROJECT_AABB_CUH
#define SPECTRAL_RT_PROJECT_AABB_CUH

#include "vec3.cuh"
#include "interval.cuh"
#include "ray.cuh"

using namespace interval;

enum Axis {
    X,
    Y,
    Z
};

class aabb {
public:

    __host__ __device__
    aabb() {} //The default AABB is empty, since intervals are empty by default

    __host__ __device__
    aabb(const numeric_interval& ix, const numeric_interval& iy, const numeric_interval& iz) : x(ix), y(iy), z(iz) {}

    __host__ __device__
    aabb(const aabb& box0, const aabb& box1) {
        x = numeric_interval(box0.x, box1.x);
        y = numeric_interval(box0.y, box1.y);
        z = numeric_interval(box0.z, box1.z);
    }
    __host__ __device__
    aabb(const point3& a, const point3& b) {
        /*
         * Treat the two points a and b as extrema for the bounding box,
         * so we don't require a particular minimum/maximum coordinate order.
         */

        x = numeric_interval(fmin(a.e[0], b.e[0]), fmax(a.e[0], b.e[0]));
        y = numeric_interval(fmin(a.e[1], b.e[1]), fmax(a.e[1], b.e[1]));
        z = numeric_interval(fmin(a.e[2], b.e[2]), fmax(a.e[2], b.e[2]));
    }

    __device__
    const numeric_interval& axis(int n) const {
        switch(n) {
            case 1:
                return y;
                break;
            case 2:
                return z;
                break;
            default:
                //Axis = X
                return x;
        }
    }

    __device__
    aabb pad() const {
        // Return an AABB that has no side narrower than some delta, padding if necessary.

        float delta = 0.0001f;
        numeric_interval new_x = (size(x.min, x.max) >= delta) ? x : expand(x.min, x.max, delta);
        numeric_interval new_y = (size(x.min, x.max) >= delta) ? y : expand(y.min, y.max, delta);
        numeric_interval new_z = (size(z.min, z.max) >= delta) ? z : expand(z.min, z.max, delta);

        return aabb(new_x, new_y, new_z);
    }

    __device__
    bool hit(const ray& r, float min, float max) const;

    numeric_interval x, y, z;

};


#endif //SPECTRAL_RT_PROJECT_AABB_CUH
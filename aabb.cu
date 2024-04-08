//
// Created by pietr on 08/04/2024.
//

#include "aabb.cuh"

__device__
bool aabb::hit(const ray &r, float min, float max) const {
    /*
         * check if a ray hits an AABB
         * for each axis check if the ray passes through the corresponding slab (i.e. intervals defined by axis(a).min and max)
         * if ray passes through all slabs then it hits the AABB
         */

    for (int a = 0; a < 3; a++) {

        auto inverseDir = 1 / r.direction()[a];
        auto orig = r.origin()[a];
        float t0, t1;

        if (inverseDir >= 0) {
            t0 = (axis(a).min - orig) * inverseDir;
            t1 = (axis(a).max - orig) * inverseDir;
        } else {
            //if direction is negative interval bounds are "reversed"
            t1 = (axis(a).min - orig) * inverseDir;
            t0 = (axis(a).max - orig) * inverseDir;
        }

        if (t0 > min)
            min = t0;
        if (t1 < max)
            max = t1;

        if (max <= min)
            return false;
    }

    return true;
}
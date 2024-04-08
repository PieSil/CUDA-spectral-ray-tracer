//
// Created by pietr on 08/04/2024.
//

#include "sphere.cuh"

__device__
bool sphere::hit(const ray &r, float min, float max, hit_record &rec) const {

    /*
     * if the ray hits the sphere returns the distance of the hit point from the origin of the ray
     * for the full math and explanation please refer to chapters 5.1, 6.2 of:
     * Ray Tracing in One Weekend by  Peter Shirley, Trevor David Black, Steve Hollasch
     * (https://raytracing.github.io/books/RayTracingInOneWeekend.html):
     */

    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = half_b*half_b - a*c;
    if (discriminant < 0.0f) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (!interval::surrounds(min, max, root)) {
        root = (-half_b + sqrtd) / a;
        if (!interval::surrounds(min, max, root))
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    // rec.mat = mat; //TODO: uncomment when adding materials

    return true;
}
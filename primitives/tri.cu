#include "tri.cuh"

__device__
bool tri::hit(const ray& r, float min, float max, hit_record& rec) const {
    /*
    * Determine whether the ray hits the plane where lies the triangle or not
    * for the full math and explanation please refer to chapters 6.2 and 6.3 of:
    * Ray Tracing: The Next Week by  Peter Shirley, Trevor David Black, Steve Hollasch
    * (https://raytracing.github.io/books/RayTracingTheNextWeek.html):
    */

    float denom = dot(normal, r.direction());

    //No hit if the ray is parallel to the plane
    if (fabs(denom) < 1e-8f) {
        return false;
    }

    //Return false if the hit point parameter t is outside the ray interval
    float t =  (D - dot(normal, r.origin())) / denom;

    if (!interval::contains(min, max, t))
        return false;

    point3 intersection = r.at(t);

    if (!is_interior_faster(intersection)) {
        return false;
    }
    
    // Ray hits the 2D shape; set the rest of the hit record and return true.

    rec.t = t;
    rec.p = intersection;
    rec.mat = mat;
    rec.set_face_normal(r, normal);

    return true;
}
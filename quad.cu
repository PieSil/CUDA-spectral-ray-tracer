//
// Created by pietr on 28/12/2023.
//

#include "quad.cuh"

bool quad::hit(const ray& r, float min, float max, hit_record& rec) const
{

    float denom = dot(normal, r.direction());

    //No hit if the ray is parallel to the plane
    if (fabs(denom) < 1e-8) {
        return false;
    }

    //Return false if the hit point parameter t is outside the ray interval
    float t = (D - dot(normal, r.origin())) / denom;
    if(!interval::contains(min, max, t))
        return false;

    /*
     * Determine the hit point lies within the planar shape using its plane coordinates.
     * for the full math and explanation please refer to chapters 6.2, 6.3, 6.4 and 6.5 of:
     * Ray Tracing: The Next Week by  Peter Shirley, Trevor David Black, Steve Hollasch
     * (https://raytracing.github.io/books/RayTracingTheNextWeek.html):
     */

    auto intersection = r.at(t);
    auto planar_hitpt_vector = intersection - Q; //vector from Q to hit point

    //A point on the plane P can be defined by P = Q + alpha*u + beta*v
    //the next lines of code compute alpha and beta using the cached value "w"
    auto alpha = dot(w, cross(planar_hitpt_vector, v));
    auto beta = dot(w, cross(u, planar_hitpt_vector));

    if (!is_interior(alpha, beta))
        return false;

    // Ray hits the 2D shape; set the rest of the hit record and return true.

    rec.t = t;
    rec.p = intersection;
    rec.mat = mat;
    rec.set_face_normal(r, normal);

    return true;
}

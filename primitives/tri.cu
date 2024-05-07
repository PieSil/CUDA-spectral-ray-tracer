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

    /*
    if (debug) {
        printf("hit!:\nrec.t: %f\nintersection: (%f, %f, %f)\nrec.normal: ()%f, %f, %f)\n\n", rec.t, rec.p[0], rec.p[1], rec.p[2], rec.normal[0], rec.normal[1], rec.normal[2]);
    }
    */

    return true;
}

__device__
void tri::init() {
    //plane implicit formula is Ax + By + Cz = - D where:
    //(A, B, C) is the plane normal
    //D is a constant
    //(x, y, z) are the coordinates of a point on the plane
    //thus we need to compute the plane normal and find D 


    vec3 n = cross(v[1] - v[0], v[2] - v[0]);
    normal = unit_vector(n);
    //printf("normal: (%f, %f, %f)\n", normal[0], normal[1], normal[2]);

    bool perp_x = fabs(dot(normal, vec3(1.f, 0.f, 0.f))) < 1e-8f;
    bool perp_y = fabs(dot(normal, vec3(0.f, 1.f, 0.f))) < 1e-8f;
    bool perp_z = fabs(dot(normal, vec3(0.f, 0.f, 1.f))) < 1e-8f;

    /* check if normal is parallel or anti-parallel to an axis and determine if an axis aligned parallel plane exists based on that */

    if (perp_y && perp_z) {
        //parallel to YZ
        aa_plane = AAPlane::YZ;
    }
    else if (perp_x && perp_z) {
        //parallel to XZ
        aa_plane = AAPlane::XZ;
    }
    else if (perp_x && perp_y) {
        //parallel to XY
        aa_plane = AAPlane::XY;
    }

    D = dot(normal, v[0]); //solves D = v1_x*n_x + v1_y*n_y + v1_z*n_z in order to find D, no need to flip the sign of D

    init_clockwise();

    set_bounding_box();
}

__device__
void tri::translate(const vec3 dir, const bool reinit) {
    v[0] += dir;
    v[1] += dir;
    v[2] += dir;

    if (reinit)
        init();
}

__device__
void tri::rotate(float theta, transform::AXIS ax, bool reinit, bool local) {
    float rot_matrix[9] = { 1.0f, 0.f, 0.f,
                0.f, 1.0f, 0.f,
                0.f, 0.f, 1.f };

    transform::assign_rot_matrix(theta, ax, rot_matrix);
    point3 center;

    if (local) {
        center = centroid();
        translate(-center, false);
    }

    v[0] = vec3::matrix_mul(v[0], rot_matrix);
    v[1] = vec3::matrix_mul(v[1], rot_matrix);
    v[2] = vec3::matrix_mul(v[2], rot_matrix);

    if (local)
        translate(center, false);

    if (reinit)
        init();
}

const bool tri::is_interior_faster(const point3 p) const {
    float a1 = double_signed_area_2D(p, v[0], v[1]);
    float a2 = double_signed_area_2D(p, v[1], v[2]);
    float a3 = double_signed_area_2D(p, v[2], v[0]);

    return clockwise ? (a1 >= 0.f && a2 >= 0.f && a3 >= 0.f) : (a1 <= 0.f && a2 <= 0.f && a3 <= 0.f);
}

__device__
const bool tri::is_interior(const point3 p) const {
    /*
     * check wether the intersection point P of the ray with the tri's plane
     * lies on the "left" side of each tri's edge
     * don't bother with early return because of divergence
     */

    vec3 edge = v[1] - v[0];
    vec3 c = p - v[0];
    bool left_edge0 = dot(normal, cross(edge, c)) >= 0;

    edge = v[2] - v[1];
    c = p - v[1];
    bool left_edge1 = dot(normal, cross(edge, c)) >= 0;

    edge = v[0] - v[2];
    c = p - v[2];
    bool left_edge2 = dot(normal, cross(edge, c)) >= 0;

    return left_edge0 && left_edge1 && left_edge2;
}

__device__
float tri::double_signed_area_2D(const point3 v1, const point3 v2, const point3 v3) const {
    /*
     * project a given triangle onto a 2D plane, then compute its double area
     * useful in order to check if vertices are given in a clockwise or counter-clockwise order
     * and also to determine whether a point p on the triangle's plane is inside or outside of the triangle
     */

    uint w_axis; //"width" axis
    uint h_axis; //"heght" axis

    switch (aa_plane) {

    case AAPlane::YZ:
        w_axis = 1;
        h_axis = 2;
        break;
    case AAPlane::XZ:
        w_axis = 0;
        h_axis = 2;
        break;

    case AAPlane::XY:
    default:
        w_axis = 0;
        h_axis = 1;
    }

    return (v1[w_axis] - v3[w_axis]) * (v2[h_axis] - v3[h_axis]) - (v2[w_axis] - v3[w_axis]) * (v1[h_axis] - v3[h_axis]);
}


#ifndef SPECTRAL_RT_PROJECT_TRI_CUH
#define SPECTRAL_RT_PROJECT_TRI_CUH

#include "hittable.cuh"
#include "materials/material.cuh"

enum AAPlane {
    NONE,  //no special trait
    XY,    //triangle lays on a plane parallel to the XY plane
    YZ,    //triangle lays on a plane parallel to the YZ plane
    XZ     //triangle lays on a plane parallel to the XZ plane
};

enum CreationMode {
    VERTICES,
    VECTORS
};

class tri : public hittable {
public:

    __device__ 
    tri(const point3 v1, const point3 v2, const point3 v3, material* m, CreationMode mode = VERTICES) : mat(m) {
        //aa_plane = AAPlane::NONE;
       
        v[0] = v1;
        switch (mode) {

        case CreationMode::VECTORS:
            v[1] = v1 + v2;
            v[2] = v1 + v3;
            break;

        case CreationMode::VERTICES:
        default:
            v[1] = v2;
            v[2] = v3;
        }

        init();
    }


    __host__ __device__
    virtual void set_bounding_box() {
        bbox = aabb(v[0], v[1], v[2]).pad();
    }

    __host__ __device__
    aabb bounding_box() const override {
        return bbox;
    }

    __device__
    bool hit(const ray& r, float min, float max, hit_record& rec) const override;

    __device__
    point3 centroid() const {

        return (v[0] + v[1] + v[2]) / 3.f;
    }

    point3 v[3];
    bool clockwise; //whether the vertices follow a clockwise or counter-clockwise order inside the v array
    AAPlane aa_plane; //signals if the triangle lies on a plane parallel to the axis
    material* mat;
    aabb bbox;
    vec3 normal;
    float D;
    //vec3 w; //cached value needed during ray intersection computation

    private:
        __device__
            void init() {
            //plane implicit formula is Ax + By + Cz = - D where:
            //(A, B, C) is the plane normal
            //D is a constant
            //(x, y, z) are the coordinates of a point on the plane
            //thus we need to compute the plane normal and find D 


            vec3 n = cross(v[1] - v[0], v[2] - v[0]);
            normal = unit_vector(n);


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
        const bool is_interior_faster(const point3 p) const {
            float a1 = double_signed_area_2D(p, v[0], v[1]);
            float a2 = double_signed_area_2D(p, v[1], v[2]);
            float a3 = double_signed_area_2D(p, v[2], v[0]);

            return clockwise ? (a1 >= 0.f && a2 >= 0.f && a3 >= 0.f) : (a1 <= 0.f && a2 <= 0.f && a3 <= 0.f);
        }

        __device__
        virtual const bool is_interior(const point3 p) const {
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
        float double_signed_area_2D(const point3 v1, const point3 v2, const point3 v3) const {
            /*
             * project a given triangle onto a 2D plane, then compute its double area
             * useful in order to check if vertices are given in a clockwise or counter-clockwise order
             * and also to determine whether a point p on the triangle's plane is inside or outside of the triangle
             */

            uint w_axis; //"width" axis
            uint h_axis; //"base" axis

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

        __device__ 
        const bool init_clockwise() {
            clockwise = double_signed_area_2D(v[0], v[1], v[2]) >= 0;
        }
};

#endif
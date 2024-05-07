

#ifndef SPECTRAL_RT_PROJECT_TRI_CUH
#define SPECTRAL_RT_PROJECT_TRI_CUH

#include "hittable.cuh"
#include "materials/material.cuh"
#include "transform.cuh"

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
    tri() {}

    __device__ 
    tri(const point3 v1, const point3 v2, const point3 v3, material* m, bool defer_init = false, CreationMode mode = VERTICES) : mat(m) {
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

        if (!defer_init)
            init();
    }

    __device__
    void init();

    __host__ __device__
    virtual void set_bounding_box() {
        bbox = aabb(v[0], v[1], v[2]).pad();
        //bbox = aabb(v[0], v[0] + v[1] - v[0] + v[2] - v[0]).pad();
    }

    __host__ __device__
    aabb bounding_box() const override {
        return bbox;
    }

    __device__
    bool hit(const ray& r, float min, float max, hit_record& rec) const override;

    __device__
    void translate(const vec3 dir, const bool reinit = true);

    __device__
    void rotate(const float theta, const transform::AXIS ax, const bool reinit = true, const bool local = true);

    __device__
    point3 centroid() const {

        return (v[0] + v[1] + v[2]) / 3.f;
    }

    __device__
    void flip_normals() {
        auto tmp = v[1];
        v[1] = v[2];
        v[2] = tmp;

        init();
    }

    point3 v[3];
    bool clockwise; //whether the vertices follow a clockwise or counter-clockwise order inside the v array
    AAPlane aa_plane; //signals if the triangle lies on a plane parallel to the axis
    material* mat;
    aabb bbox;
    vec3 normal;
    //bool debug = false;
    float D;

private:

    __device__
    const bool is_interior_faster(const point3 p) const;

    __device__
    const bool is_interior(const point3 p) const;

    __device__
        float double_signed_area_2D(const point3 v1, const point3 v2, const point3 v3) const;

    __device__ 
    const bool init_clockwise() {
        clockwise = double_signed_area_2D(v[0], v[1], v[2]) >= 0;
    }
};

#endif
//
// Created by pietr on 23/11/2023.
//

#ifndef RTWEEKEND_CUDA_CAMERA_BUILDER_CUH
#define RTWEEKEND_CUDA_CAMERA_BUILDER_CUH

#include "camera.cuh"
#include "params.h"

class camera_builder {
public:

    __host__
    camera_builder setVfov(float _vfov) {
        vfov = _vfov;
        return *this;
    }

    __host__
    camera_builder setLookfrom(point3 _lookfrom) {
        lookfrom = _lookfrom;
        return *this;
    }

    __host__
    camera_builder setLookat(point3 _lookat) {
        lookat = _lookat;
        return *this;
    }

    __host__
    camera_builder setVup(vec3 _vup) {
        vup = _vup;
        return *this;
    }

    __host__
    camera_builder setDefocusAngle(float _da) {
        defocus_angle = _da;
        return *this;
    }

    __host__
    camera_builder setFocusDist(float _fd) {
        focus_dist = _fd;
        return *this;
    }

    camera_builder setBackground(color c) {
        background = c;
        return *this;
    }

    __host__
    camera getCamera() {
        auto pm = param_manager::getInstance();
        return camera(pm->getParams().getAR(), pm->getParams().getXres(), pm->getParams().getYres(), vfov, lookfrom, lookat, vup, defocus_angle, focus_dist, background);
    }

private:
    float vfov = 90.0f;
    point3 lookfrom = point3(0,0,-1);
    point3 lookat = point3(0,0,0);
    vec3   vup = vec3(0,1,0);
    float defocus_angle = 0;  // Variation angle of rays through each pixel
    float focus_dist = 10;
    color background = color(0, 0, 0);
};

#endif //RTWEEKEND_CUDA_CAMERA_BUILDER_CUH

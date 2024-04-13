//
// Created by pietr on 13/04/2024.
//

#ifndef SPECTRAL_RT_PROJECT_CAMERA_CUH
#define SPECTRAL_RT_PROJECT_CAMERA_CUH

#include "rendering.cuh"

class camera {

public:

    __host__
    camera(float ar, int w, float _vfov, point3 _lookfrom, point3 _lookat, vec3 _vup, float _da, float _fd, color bg)
            : aspect_ratio(ar), image_width(w), vfov(_vfov), lookfrom(_lookfrom),
              lookat(_lookat),vup(_vup), defocus_angle(_da), focus_dist(_fd), background(bg) {
        initialize();
    }

    __host__
    camera() {};

    __host__
    void render(bvh *bvh, uint bounce_limit, uint samples_per_pixel) const;

    __host__
    void render(bvh *bvh, uint bounce_limit, uint samples_per_pixel, dim3 blocks, dim3 threads) const;

    //getters

    __host__
    const int& getImageWidth() const {
        return image_width;
    }

    __host__
    const int& getImageHeight() const {
        return image_height;
    }

    __host__
    const uint& getNumPixels() const {
        return num_pixels;
    }

    __host__
    const point3& getCenter() const {
        return center;
    }

    __host__
    const point3& getPixel00Loc() const {
        return pixel00_loc;
    }

    __host__
    const vec3& getPixelDeltaU() const {
        return pixel_delta_u;
    }

    __host__
    const vec3& getPixelDeltaV() const {
        return pixel_delta_v;
    }

    float getDefocusAngle() const {
        return defocus_angle;
    }

    const vec3 &getDefocusDiskU() const {
        return defocus_disk_u;
    }

    const vec3 &getDefocusDiskV() const {
        return defocus_disk_v;
    }

    const color &getBackground() const {
        return background;
    }

private:
    float aspect_ratio; //Ratio of image width over height
    int image_width; //Rendered image width in pixel count
    int image_height; //Rendered image height in pixel count
    float vfov; //Vertical view angle (field of view)

    point3 lookfrom;  // Point camera is looking from
    point3 lookat;   // Point camera is looking at
    vec3   vup;     // Camera-relative "up" direction

    float defocus_angle;  //angle of the cone with apex at the viewport center and base at the camera center
    //(cone base will be our "defocus disk")
    float focus_dist;  //distance from camera to point of perfect focus

    uint num_pixels; //Total number of pixels that make up the image
    point3 center; //Camera center
    point3 pixel00_loc; //Location of pixel 0, 0
    vec3 pixel_delta_u; //Offset to pixel to the right
    vec3 pixel_delta_v; //Offset to pixel below
    vec3   u, v, w; //camera frame basis vectors
    //u: camera "right" direction
    //v: camera "up" direction
    //w: opposite of camera "front" direction (-w = camera "front" direction)

    vec3   defocus_disk_u;  // Defocus disk horizontal radius
    vec3   defocus_disk_v;  // Defocus disk vertical radius
    color background;       // Scene background color;

    __host__
    void initialize();

};

#endif //SPECTRAL_RT_PROJECT_CAMERA_CUH

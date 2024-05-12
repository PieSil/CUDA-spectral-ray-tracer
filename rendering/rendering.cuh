//
// Created by pietr on 09/04/2024.
//

#ifndef SPECTRAL_RT_PROJECT_RENDERING_CUH
#define SPECTRAL_RT_PROJECT_RENDERING_CUH

#include "cuda_utility.cuh"
#include "color.cuh"
#include "hittable.cuh"
#include "bvh.cuh"
#include "io.cuh"
#include "camera.cuh"
#include "frame_buffer.cuh"
#include "material.cuh"

struct camera_data {

    camera_data(uint w, uint h, vec3 delta_u, vec3 delta_v, point3 p00_loc,float def_angle, point3 center, vec3 def_disk_u,
             vec3 def_disk_v) : width(w), height(h), pixel_delta_u(delta_u), pixel_delta_v(delta_v),
             pixel00_loc(p00_loc),  defocus_angle(def_angle), camera_center(center), defocus_disk_u(def_disk_u),
             defocus_disk_v(def_disk_v){};

    uint width;
    uint height;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    point3 pixel00_loc;
    float defocus_angle;
    point3 camera_center;
    vec3 defocus_disk_u;
    vec3 defocus_disk_v;
};

class camera;

__device__
void ray_bounce(ray &r, const float *background_emittance_spectrum, bvh** bvh, uint bounce_limit, curandState *local_rand_state);

__device__
ray get_ray(uint i, uint j, const point3 pixel00_loc, const vec3 pixel_delta_u, const vec3 pixel_delta_v,
            const point3 camera_center, const vec3 defocus_disk_u, const vec3 defocus_disk_v, const float defocus_angle,
            curandState *local_rand_state);

__device__
ray get_ray_stratified_sample(uint i, uint j,
                              point3 pixel00_loc,
                              vec3 pixel_delta_u,
                              vec3 pixel_delta_v,
                              uint sample_x,
                              uint sample_y,
                              float recip_sqrt_spp,
                              point3 camera_center,
                              float defocus_angle,
                              vec3 defocus_disk_u,
                              vec3 defocus_disk_v,
                              curandState* local_rand_state);

__device__
vec3 pixel_sample_square(vec3 pixel_delta_u, vec3 pixel_delta_v, curandState* local_rand_state);

__device__
vec3 pixel_stratified_sample_square(uint sample_x, uint sample_y, float recip_sqrt_spp, vec3 pixel_delta_u, vec3 pixel_delta_v, curandState* local_rand_state);

__device__
point3 defocus_disk_sample(vec3 camera_center, vec3 defocus_disk_u, vec3 defocus_disk_v, curandState* local_rand_state);

void call_render_kernel(frame_buffer* fb, bvh **bvh, uint samples_per_pixel, const camera *cam, uint bounce_limit,
                        dim3 blocks, dim3 threads);

#endif //SPECTRAL_RT_PROJECT_RENDERING_CUH

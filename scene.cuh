//
// Created by pietr on 13/04/2024.
//

#ifndef SPECTRAL_RT_PROJECT_SCENE_CUH
#define SPECTRAL_RT_PROJECT_SCENE_CUH

#define HALF_N_RANDOM_I 11
#define HALF_N_RANDOM_J 11

#define MATERIAL_GROUND_ID 0
#define MATERIAL_CENTER_ID 1
#define MATERIAL_LEFT_ID 2
#define MATERIAL_RIGHT_ID 3
#define N_RANDOM_MATERIALS (2*HALF_N_RANDOM_I * 2*HALF_N_RANDOM_J)
#define N_NON_RANDOM_MATERIALS 4
#define RANDOM_WORLD_MATERIALS (N_RANDOM_MATERIALS + N_NON_RANDOM_MATERIALS)

#define N_RANDOM_SPHERES (2*HALF_N_RANDOM_I * 2*HALF_N_RANDOM_J)
#define N_NON_RANDOM_SPHERES 4
#define N_SPHERES (N_RANDOM_SPHERES + N_NON_RANDOM_SPHERES)
#define RANDOM_WORLD_SIZE N_SPHERES

#define WORLD_SELECTOR 3

#include "cuda_utility.cuh"
#include "bvh.cuh"
#include "sphere.cuh"
#include "quad.cuh"
#include "camera_builder.cuh"
#include "materials/material.cuh"
#include "transform.cuh"
#include "sellmeier.cuh"

namespace scene {
    __device__
    void device_random_world(hittable **d_list, material **d_mat_list, int *world_size, int *n_materials);

    __device__
    void device_quad_world(hittable **d_list, material **d_mat_list);

    __device__
    void device_simple_light(hittable **d_list, material **d_mat_list);

    __device__
    void device_cornell_box(hittable **d_list, material **d_mat_list);

    __device__
    void device_3_spheres(hittable** d_list, material** d_mat_list);

    __host__
    void init_world_parameters(uint world_selector, int *world_size_ptr, int *n_materials_ptr);

    __host__
    camera_builder random_world_cam_builder();

    __host__
    camera_builder quad_world_camera_builder();

    __host__
    camera_builder simple_light_camera_builder();

    __host__
    camera_builder cornell_box_camera_builder();

    __host__
    camera_builder spheres_camera_builder();

    __host__
    bool create_bvh(hittable **d_world, size_t world_size, bvh **d_bvh);

    __host__
    void create_world(hittable **d_list, material **d_mat_list, int *world_size, int *n_materials,
                      float *dev_sRGBToSpectrum_Data);

    __host__
    void free_world(hittable **d_list, bvh **dev_bvh, material **d_mat_list, int world_size,
                    int n_materials);

    struct result {
        bool success;
        string msg;
    };

    class scene_manager {
    public:
        __host__
        explicit scene_manager() {
            cam_inited = false;
            world_inited = false;
            init_camera();
            world_result = init_world();
        };

        __host__
        virtual ~scene_manager() {
            destroy_world();
        }

        int *h_n_materials_ptr = new int;
        int *h_world_size_ptr = new int;

        void render(uint bounce_limit, uint samples_per_pixel) {
            if (cam_inited && world_inited)
                cam.render(dev_bvh, bounce_limit, samples_per_pixel);
        }

        [[nodiscard]] result getResult() const {
            return world_result;
        }

    private:
        __host__
        const result init_world();

        __host__
        void init_camera();

        __host__
        void destroy_world();

        hittable **dev_world;
        material **dev_mat_list;
        bvh **dev_bvh;

        camera cam;
        bool cam_inited;
        bool world_inited;

        result world_result;
    };
}

#endif //SPECTRAL_RT_PROJECT_SCENE_CUH

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

#define WORLD_SELECTOR 6

#include "cuda_utility.cuh"
#include "bvh.cuh"
#include "sphere.cuh"
#include "quad.cuh"
#include "camera_builder.cuh"
#include "material.cuh"
#include "transform.cuh"
#include "sellmeier.cuh"
#include "tri.cuh"
#include "tri_quad.cuh"
#include "prism.cuh"
#include "tri_box.cuh"
#include "rendering.cuh"

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
    void device_prism_test(hittable** d_list, material** d_mat_list);

    __device__
    void device_tri_world(hittable** d_list, material** d_mat_list);

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
    camera_builder prism_test_camera_builder();

    __host__
    camera_builder tris_camera_builder();

    __host__
    camera_builder spheres_camera_builder();

    __host__
    bool create_bvh(hittable **d_world, size_t world_size, bvh **d_bvh);

    __host__
    void create_world(hittable **d_list, material **d_mat_list, int *world_size, int *n_materials, float* dev_sRGBToSpectrum_Data);

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

        void render() {
            if (renderer_inited)
                r.call_render_kernel();
            else
                cerr << "Renderer not yet initialized";
        }

        void init_renderer(frame_buffer* fb, uint bounce_limit, uint samples_per_pixel) {
            if (cam_inited && world_inited) {
                r = renderer(fb, dev_bvh, samples_per_pixel, &cam, bounce_limit);
                renderer_inited = true;
            }

            else {
                if (!cam_inited)
                    cerr << "Camera not yet initialized" << endl;
                if(!world_inited)
                    cerr << "World not yet initialized" << endl;
            }
        }

        void init_device_params(uint threads, uint blocks, uint chunk_width, uint chunk_height) {
            if (renderer_inited) {
                r.init_device_params(threads, blocks, chunk_width, chunk_height);
            } else 
                cerr << "Init renderer before assigning device parameters" << endl;
        }

        void init_device_params(uint chunk_width, uint chunk_height) {
            if (renderer_inited) {
                uint tx = 8;
                uint ty = 8;

                dim3 blocks(cam.getImageWidth() / tx + 1, cam.getImageHeight() / ty + 1);
                dim3 threads(tx, ty);
                r.init_device_params(threads, blocks, chunk_width, chunk_height);
            }
            else
                cerr << "Init renderer before assigning device parameters" << endl;
        }

        void init_device_params() {
            if (renderer_inited) {
                uint width = cam.getImageWidth();
                uint height = cam.getImageHeight();
                uint tx = 8;
                uint ty = 8;

                dim3 blocks(width / tx + 1, height / ty + 1);
                dim3 threads(tx, ty);
                r.init_device_params(threads, blocks, width, height);
            }
            else
                cerr << "Init renderer before assigning device parameters" << endl;
        }

        [[nodiscard]] result getResult() const {
            return world_result;
        }

        const uint img_width() const {
            return cam.getImageWidth();
        }

        const uint img_height() const {
            return cam.getImageHeight();
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
        bool cam_inited = false;
        bool world_inited = false;
        renderer r;
        bool renderer_inited = false;

        result world_result;
    };
}

#endif //SPECTRAL_RT_PROJECT_SCENE_CUH

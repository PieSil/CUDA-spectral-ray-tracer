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

#include "cuda_utility.cuh"
#include "bvh.cuh"
#include "camera_builder.cuh"
#include "material.cuh"
#include "transform.cuh"
#include "sellmeier.cuh"
#include "tri.cuh"
#include "tri_quad.cuh"
#include "prism.cuh"
#include "tri_box.cuh"
#include "pyramid.cuh"
#include "rendering.cuh"
#include "log_context.h"
#include "params.h"

namespace scene {

	__device__
		void device_test_scene(tri** d_list, material** d_mat_list);

	__device__
		void device_prism_test(tri** d_list, material** d_mat_list);

	__device__
		void device_different_mats_world(tri** d_list, material** d_mat_list);

	__device__
		void device_orig_cornell_box_scene(tri** d_list, material** d_mat_list);

	__host__
		void init_world_parameters(uint world_selector, int* world_size_ptr, int* n_materials_ptr);

	__host__
		camera_builder test_scene_camera_builder();

	__host__
		camera_builder prism_test_camera_builder();

	
	__host__
		camera_builder different_mats_camera_builder();

	__host__
		camera_builder original_cornell_box_camera_builder();

	__host__
		bool create_bvh(tri** d_world, size_t world_size, bvh** d_bvh);

	__host__
		void create_world(uint selected_world, tri** d_list, material** d_mat_list, int* world_size, int* n_materials, float* dev_sRGBToSpectrum_Data);

	__host__
		void free_world(tri** d_list, bvh** dev_bvh, material** d_mat_list, int world_size,
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
			selected_world = param_manager::getInstance()->getParams().getSceneId();
			init_camera();
			world_result = init_world();
		};

		__host__
			virtual ~scene_manager() {
			destroy_world();
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

		bvh** getWorld() {
			if (world_inited)
				return dev_bvh;
			else
				return nullptr;
		}

		camera* getCamPtr() {
			if (cam_inited)
				return &cam;
			else
				return nullptr;
		}


	private:
		__host__
			const result init_world();

		__host__
			void init_camera();

		__host__
			void destroy_world();

		int* h_n_materials_ptr = new int;
		int* h_world_size_ptr = new int;

		tri** dev_world;
		material** dev_mat_list;
		bvh** dev_bvh;

		camera cam;
		bool cam_inited = false;
		bool world_inited = false;

		uint selected_world;
		result world_result;
	};
}

#endif //SPECTRAL_RT_PROJECT_SCENE_CUH

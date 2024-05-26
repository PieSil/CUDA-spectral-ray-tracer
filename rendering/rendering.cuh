//
// Created by pietr on 09/04/2024.
//

#ifndef SPECTRAL_RT_PROJECT_RENDERING_CUH
#define SPECTRAL_RT_PROJECT_RENDERING_CUH

#include "cuda_utility.cuh"
#include "color.cuh"
#include "bvh.cuh"
#include "io.cuh"
#include "camera.cuh"
#include "frame_buffer.cuh"
#include "material.cuh"
#include "camera.cuh"
#include "log_context.h"

struct camera_data {

    camera_data() {};

	camera_data(uint w, uint h, vec3 delta_u, vec3 delta_v, point3 p00_loc, float def_angle, point3 center, vec3 def_disk_u,
		vec3 def_disk_v) : width(w), height(h), pixel_delta_u(delta_u), pixel_delta_v(delta_v),
		pixel00_loc(p00_loc), defocus_angle(def_angle), camera_center(center), defocus_disk_u(def_disk_u),
		defocus_disk_v(def_disk_v) {};

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

class renderer {

public:

	__host__
	renderer() {};

	__host__
	renderer(bvh** _bvh, uint _samples_per_pixel, camera* cam, uint _bounce_limit) :
		dev_bvh(_bvh), samples_per_pixel(_samples_per_pixel), bounce_limit(_bounce_limit), background(cam->getBackground()) {
		assign_cam_data(cam);

	}

	~renderer() {
		if (device_inited) {
			checkCudaErrors(cudaFree(dev_rand_state));
			checkCudaErrors(cudaFree(dev_fb));
			checkCudaErrors(cudaFree(dev_background_spectrum));
			checkCudaErrors(cudaGetLastError());
		}
	}

	void render(uint offset_x, uint offset_y) {

		//no check on parameters validity for minimum overhead
		call_render_kernel(max_chunk_width, max_chunk_height, offset_x, offset_y);
	}

	void render(uint width, uint height, uint offset_x, uint offset_y) {
		//no check on parameters validity for minimum overhead
		call_render_kernel(width, height, offset_x, offset_y);
	}

	__host__
    void init_device_params(dim3 _threads, dim3 _blocks, uint _max_chunk_width, uint _max_chunk_height);

	__device__
		static void ray_bounce(ray& r, const float* background_emittance_spectrum, bvh** bvh, uint bounce_limit, bvh_node* node_cache, curandState* local_rand_state);

	__device__
		static ray get_ray(uint i, uint j, const point3 pixel00_loc, const vec3 pixel_delta_u, const vec3 pixel_delta_v,
			const point3 camera_center, const vec3 defocus_disk_u, const vec3 defocus_disk_v, const float defocus_angle,
			curandState* local_rand_state);

	const uint getMaxChunkWidth() const {
		return max_chunk_width;
	}

	const uint getMaxChunkHeight() const {
		return max_chunk_height;
	}

	const vec3* getDevFB() const {
		return dev_fb;
	}

private:
	__host__
	void call_render_kernel(uint width, uint height, uint offset_x, uint offset_y);

	__host__
	void assign_cam_data(camera* cam);

	__device__
		static ray get_ray_stratified_sample(uint i, uint j,
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
		static vec3 pixel_sample_square(vec3 pixel_delta_u, vec3 pixel_delta_v, curandState* local_rand_state);

	__device__
		static vec3 pixel_stratified_sample_square(uint sample_x, uint sample_y, float recip_sqrt_spp, vec3 pixel_delta_u, vec3 pixel_delta_v, curandState* local_rand_state);

	__device__
		static point3 defocus_disk_sample(vec3 camera_center, vec3 defocus_disk_u, vec3 defocus_disk_v, curandState* local_rand_state);

    camera_data cam_data;
    bvh** dev_bvh;
    uint samples_per_pixel;
    uint bounce_limit;
	color background;

	uint max_chunk_width;
	uint max_chunk_height;
    dim3 blocks;
    dim3 threads;
    uint shared_mem_size;
    vec3* dev_fb = nullptr;
    curandState* dev_rand_state = nullptr;

    float* dev_background_spectrum = nullptr;

    bool device_inited = false;
};

__host__ __device__
inline void map_block_idxs(uint& row, uint& col, uint matrix_row, uint matrix_col, uint block_row, uint block_col, uint block_width, uint block_height, uint matrix_width, uint matrix_height) {
	row = matrix_row + block_row;
	col = matrix_col + block_col;

	//bool overflow_x = false;
	//bool overflow_y = false;
	bool overflow_x = col >= matrix_width;
	bool overflow_y = !overflow_x && col >= matrix_height;
	uint x_overflow_size = 0;
	uint y_overflow_size = 0;

		if (overflow_x) {
			//overflow along x
			x_overflow_size = col - matrix_width;
			row += block_height;
			col = x_overflow_size;
			overflow_x = false;
			overflow_y = row >= matrix_height;

			if (overflow_y) {
				y_overflow_size = row - matrix_height;
			}
		}
		else {
			//overflow along y
			y_overflow_size = row - matrix_height;
			col += block_width;
			row = y_overflow_size;
			overflow_y = false;
			overflow_x = col >= matrix_width;
		}

}

#endif //SPECTRAL_RT_PROJECT_RENDERING_CUH

#ifndef SPECTRAL_RT_PROJECT_RENDER_MANAGER_CUH
#define SPECTRAL_RT_PROJECT_RENDER_MANAGER_CUH

#include "camera.cuh"
#include "rendering.cuh"
#include "multithread.cuh"
#include "log_context.h"

struct render_step_data {

	render_step_data() : empty(1), full(0) {}

	void alloc_buffer(size_t buffer_size) {
		fb_r = new float[buffer_size];
		fb_g = new float[buffer_size];
		fb_b = new float[buffer_size];
	}

	~render_step_data() {
		delete[] fb_r;
		delete[] fb_g;
		delete[] fb_b;
	}

	float* fb_r = nullptr;
	float* fb_g = nullptr;
	float* fb_b = nullptr;
	uint starting_offset_x = 0;
	uint starting_offset_y = 0;
	uint chunk_width = 0;
	uint chunk_height = 0;
	binary_semaphore empty;
	binary_semaphore full;
	bool is_last = false;
};

class render_manager {
public:

	render_manager(bvh** _dev_bvh, camera* _cam, frame_buffer* _fb) {
		if (_dev_bvh != nullptr && _cam != nullptr && _fb != nullptr) {
			dev_bvh = _dev_bvh;
			cam = _cam;
			fb_r = _fb->r;
			fb_g = _fb->g;
			fb_b = _fb->b;
			image_width = cam->getImageWidth();
			image_height = cam->getImageHeight();
			scene_inited = true;
		}
	}

	~render_manager() {
		end_render();
	}

	bool step();

	void init_renderer(uint bounce_limit, uint samples_per_pixel);

	void init_device_params(dim3 threads, dim3 blocks, uint _chunk_width, uint _chunk_height);

	void init_device_params(uint _chunk_width, uint _chunk_height);

	void init_device_params();

	bool update_fb() {

		bool last_read;

		render_step_data* render_data = &render_data_container[next_read_render_data_index];
		uint index = next_read_render_data_index;
		next_read_render_data_index = (next_read_render_data_index + 1) % 2;

		render_data->full.acquire();

		size_t n_rows = render_data->chunk_height;
		size_t n_cols = render_data->chunk_width;
		size_t offs_x = render_data->starting_offset_x;
		size_t offs_y = render_data->starting_offset_y;
		float* tmp_fb_r = render_data->fb_r;
		float* tmp_fb_g = render_data->fb_g;
		float* tmp_fb_b = render_data->fb_b;

		uint block_size = threads.x * threads.y;
		uint grid_size = blocks.x * blocks.y;
		for (uint idx = 0; idx < block_size * grid_size; idx++) {
			//compute index of the block that wrote this pixel
			uint blockIdx = idx / block_size;

			//get 2D indices of block
			uint block_x = blockIdx % blocks.x;
			uint block_y = blockIdx / blocks.x;

			//compute specific index of thread that wrote this pixel
			uint thread_idx = idx % block_size;

			//get 2D indices of thread
			uint thread_x = thread_idx % threads.x;
			uint thread_y = thread_idx / threads.x;

			//start from first pixel
			uint fb_x = 0;
			uint fb_y = 0;

			//move right by the number of current block
			fb_x += threads.x * block_x;

			//sum thread idx along x
			fb_x += thread_x;

			//move down by the number of current block
			fb_y += threads.y * block_y;

			//sum thread idx along y
			fb_y += thread_y;

			//ensure current 2D pixel indices fit within the current chunk
			if (fb_x < n_cols && fb_y < n_rows) {

				//sum chunk offset
				fb_x += offs_x;
				fb_y += offs_y;

				//linear access to frame buffer to store color
				size_t fb_pixel_index = fb_y * cam->getImageWidth() + fb_x;

				fb_r[fb_pixel_index] = tmp_fb_r[idx];
				fb_g[fb_pixel_index] = tmp_fb_g[idx];
				fb_b[fb_pixel_index] = tmp_fb_b[idx];
			}



		}

		last_read = render_data->is_last;
  		render_data->empty.release();

		return !last_read;
	}

	const bool isDone() const {
		return done;
	}

	const uint getImWidth() const {
		return image_width;
	}

	const uint getImHeight() const {
		return image_height;
	}

	const bool isReadyToRender() const {
		return (device_inited && i < n_iterations);
	}

	void render_cycle() {
		end_render();

		done = false;
		render_worker = thread(&render_manager::render_loop, this);
		worker_started = true;
	}

	void end_render() {
		if (worker_started) {
			render_worker.join();
			worker_started = false;
		}
	}

private:

	void render_loop() {
		while (step()) {
			;
		}
	}

    //validity flags
    bool scene_inited = false;
    bool device_inited = false;

    //scene
	bvh** dev_bvh;
	camera* cam;
	uint image_width;
	uint image_height;

    //render
	renderer r;
	bool renderer_inited = false;
	float* fb_r;
	float* fb_g;
	float* fb_b;

    //multithreading
    thread render_worker;
	bool worker_started = false;
    render_step_data render_data_container[2];
    size_t next_write_render_data_index = 0;
    size_t next_read_render_data_index = 0;
	bool done = true;
	binary_semaphore is_done_sem{ 1 };

    //image chunks management
	uint i = 0;
	uint chunk_width;
	uint chunk_height;
	uint n_iterations;
	uint x_chunks;
	uint offset_x = 0;
	uint offset_y = 0;
	uint last_offset_x = 0;
	uint last_offset_y = 0;
	uint last_chunk_width = 0;
	uint last_chunk_height = 0;

    //device parameters
	dim3 threads;
	dim3 blocks;
};

#endif
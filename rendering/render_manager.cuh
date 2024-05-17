#ifndef SPECTRAL_RT_PROJECT_RENDER_MANAGER_CUH
#define SPECTRAL_RT_PROJECT_RENDER_MANAGER_CUH

#include "camera.cuh"
#include "rendering.cuh"
#include "multithread.cuh"

struct render_step_data {

	render_step_data() : empty(1), full(0) {}

	void alloc_buffer(size_t buffer_size) {
		fb = new vec3[buffer_size];
	}

	~render_step_data() {
		if (fb != nullptr) {
			delete[] fb;
		}
	}

	vec3* fb = nullptr;
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

	render_manager(bvh** _dev_bvh, camera* _cam, vec3* _fb) {
		if (_dev_bvh != nullptr && _cam != nullptr && _fb != nullptr) {
			dev_bvh = _dev_bvh;
			cam = _cam;
			fb = _fb;
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
		vec3* tmp_fb = render_data->fb;

		for (size_t row = 0; row < n_rows; row++) {
			for (size_t col = 0; col < n_cols; col++) {
				size_t tmp_pixel_index = row * n_cols + col;
				size_t fb_pixel_index = (row + offs_y) * cam->getImageWidth() + (col + offs_x);
				fb[fb_pixel_index] = tmp_fb[tmp_pixel_index];
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

	void start_render() {
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

	bvh** dev_bvh;
	camera* cam;
	uint image_width;
	uint image_height;
	bool scene_inited = false;

	renderer r;
	bool renderer_inited = false;
	vec3* fb;
	render_step_data render_data_container[2];
	size_t next_write_render_data_index = 0;
	size_t next_read_render_data_index = 0;

	bool worker_started = false;
	bool done = true;
	binary_semaphore is_done_sem{ 1 };
	//vec3* tmp_fb;

	bool device_inited = false;

	uint i = 0;
	uint chunk_width;
	uint chunk_height;
	uint n_iterations;
	//uint chunk_size;
	uint x_chunks;
	//uint y_chunks;

	uint offset_x = 0;
	uint offset_y = 0;
	uint last_offset_x = 0;
	uint last_offset_y = 0;
	uint last_chunk_width = 0;
	uint last_chunk_height = 0;
	thread render_worker;
};

#endif
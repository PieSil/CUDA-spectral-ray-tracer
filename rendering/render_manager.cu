#include "render_manager.cuh"

bool render_manager::step() {

	//read and update shared data

	uint endpoint_x = chunk_width + offset_x;
	uint endpoint_y = chunk_height + offset_y;
	bool last_step = false;

	if (endpoint_x > image_width) {
		last_chunk_width = chunk_width - (endpoint_x - image_width);
	}
	else {
		last_chunk_width = chunk_width;

	}

	if (endpoint_y > image_height) {
		last_chunk_height = chunk_height - (endpoint_y - image_height);
	}
	else {
		last_chunk_height = chunk_height;
	}

	//get next render data object
	render_step_data* render_data = &render_data_container[next_write_render_data_index];
	uint index = next_write_render_data_index;
	next_write_render_data_index = (next_write_render_data_index + 1) % 2;

	render_data->empty.acquire(); //acquire write access on render data object

	render_data->chunk_width = last_chunk_width;
	render_data->chunk_height = last_chunk_height;

	render_data->starting_offset_x = offset_x;
	render_data->starting_offset_y = offset_y;

	r.render(last_chunk_width, last_chunk_height, offset_x, offset_y);

	size_t size = threads.x * blocks.x * threads.y * blocks.y * sizeof(float);

	checkCudaErrors(cudaMemcpyAsync(render_data->fb_r, r.getDevFBr(), size/*chunk_width * chunk_height * sizeof(float)*/, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyAsync(render_data->fb_g, r.getDevFBg(), size/*chunk_width * chunk_height * sizeof(float)*/, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyAsync(render_data->fb_b, r.getDevFBb(), size/*chunk_width * chunk_height * sizeof(float)*/, cudaMemcpyDeviceToHost));

	i++;
	if (i == n_iterations) {
		render_data->is_last = true;
		last_step = true;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	render_data->full.release(); //grant read access to produced render data

	last_offset_x = offset_x;
	last_offset_y = offset_y;

	uint n_row = i / x_chunks;
	uint n_col = i % x_chunks;

	offset_x = n_col * chunk_width;
	offset_y = n_row * chunk_height;

	return !last_step;
}

void render_manager::init_device_params(dim3 threads, dim3 blocks, uint _chunk_width, uint _chunk_height) {
	if (renderer_inited) {
		chunk_width = _chunk_width;
		chunk_height = _chunk_height;

		x_chunks = ceil(float(cam->getImageWidth()) / float(chunk_width));
		uint y_chunks = ceil(float(cam->getImageHeight()) / float(chunk_height));
		n_iterations = x_chunks * y_chunks;

		r.init_device_params(threads, blocks, chunk_width, chunk_height);
		device_inited = true;

		render_data_container[0].alloc_buffer(threads.x * blocks.x * threads.y * blocks.y);
		render_data_container[1].alloc_buffer(threads.x * blocks.x * threads.y * blocks.y);

		auto lc = log_context::getInstance();
	}
	else
		cerr << "Initialize renderer before assigning device parameters" << endl;
}

void render_manager::init_device_params(uint _chunk_width, uint _chunk_height) {
	if (renderer_inited) {
		uint tx = 16;
		uint ty = 16;

		blocks = dim3(_chunk_width / tx + 1, _chunk_height / ty + 1);
		threads = dim3(tx, ty);
		this->init_device_params(threads, blocks, _chunk_width, _chunk_height);
	}
	else
		cerr << "Init renderer before assigning device parameters" << endl;
}

void render_manager::init_device_params() {

	if (renderer_inited) {
		uint width = cam->getImageWidth();
		uint height = cam->getImageHeight();

		uint tx = 16;
		uint ty = 16;

		blocks = dim3(width / tx + 1, height / ty + 1);
		threads = dim3(tx, ty);
		this->init_device_params(threads, blocks, width, height);
	}
	else
		cerr << "Init renderer before assigning device parameters" << endl;
}

void render_manager::init_renderer(uint bounce_limit, uint samples_per_pixel) {
	if (scene_inited) {
		r = renderer(dev_bvh, samples_per_pixel, cam, bounce_limit);
		auto lc = log_context::getInstance();
		lc->add_entry("samples per pixel", samples_per_pixel);
		lc->add_entry("bounce limit", bounce_limit);

		renderer_inited = true;
	}
	else {
		cerr << "Scene not yet initialized" << endl;
	}
}
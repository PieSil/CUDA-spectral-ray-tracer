#include "render_manager.cuh"

void render_manager::step() {
	if (device_inited && i < n_iterations) {


		/*
		if (i == n_iterations - 1) {
			uint endpoint_x = chunk_width + offset_x;
			uint endpoint_y = chunk_height + offset_y;
			last_chunk_width = chunk_width - (endpoint_x - image_width);
			last_chunk_height = chunk_height - (endpoint_y - image_height);
		}
		else {
			last_chunk_width = chunk_width;
			last_chunk_height = chunk_height;
		}
		*/


		uint endpoint_x = chunk_width + offset_x;
		uint endpoint_y = chunk_height + offset_y;

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


		//uint linear_offset = (cam->getImageWidth() % chunk_width); //linear index of the first pixel of the next chunk to render
		r.render(last_chunk_width, last_chunk_height, offset_x, offset_y);

		checkCudaErrors(cudaMemcpy(tmp_fb, r.getDevFB(), chunk_width * chunk_height * sizeof(vec3), cudaMemcpyDeviceToHost));

		i++;


		last_offset_x = offset_x;
		last_offset_y = offset_y;

		/*
		offset_x += last_chunk_width;
		if (offset_x >= image_width) {
			offset_x -= image_width;

			offset_y += last_chunk_height;
			if (offset_y >= image_height) {
				offset_y -= image_height;
			}
		}
		*/

		uint n_row = i / x_chunks;
		uint n_col = i % x_chunks;

		last_offset_x = offset_x;
		last_offset_y = offset_y;

		offset_x = n_col * chunk_width;
		offset_y = n_row * chunk_height;



		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
		cerr << "Device parameters not yet initialized";
}

void render_manager::init_device_params(dim3 threads, dim3 blocks, uint _chunk_width, uint _chunk_height) {
	if (renderer_inited) {
		chunk_width = _chunk_width;
		chunk_height = _chunk_height;

		uint chunk_size = chunk_width * chunk_height;

		x_chunks = ceil(float(cam->getImageWidth()) / float(chunk_width));
		uint y_chunks = ceil(float(cam->getImageHeight()) / float(chunk_height));
		n_iterations = x_chunks * y_chunks;

		r.init_device_params(threads, blocks, chunk_width, chunk_height);
		device_inited = true;

		tmp_fb = new vec3[chunk_size];
	}
	else
		cerr << "Initialize renderer before assigning device parameters" << endl;
}

void render_manager::init_device_params(uint _chunk_width, uint _chunk_height) {
	if (renderer_inited) {
		uint tx = 8;
		uint ty = 8;

		dim3 blocks(_chunk_width / tx + 1, _chunk_height / ty + 1);
		dim3 threads(tx, ty);
		this->init_device_params(threads, blocks, _chunk_width, _chunk_height);
	}
	else
		cerr << "Init renderer before assigning device parameters" << endl;
}

void render_manager::init_device_params() {

	if (renderer_inited) {
		uint width = cam->getImageWidth();
		uint height = cam->getImageHeight();

		uint tx = 8;
		uint ty = 8;

		dim3 blocks(width / tx + 1, height / ty + 1);
		dim3 threads(tx, ty);
		this->init_device_params(threads, blocks, width, height);
	}
	else
		cerr << "Init renderer before assigning device parameters" << endl;
}

void render_manager::init_renderer(uint bounce_limit, uint samples_per_pixel) {
	if (scene_inited) {
		r = renderer(dev_bvh, samples_per_pixel, cam, bounce_limit);
		renderer_inited = true;
	}
	else {
		cerr << "Scene not yet initialized" << endl;
	}
}
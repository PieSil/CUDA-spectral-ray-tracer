//
// Created by pietr on 09/04/2024.
//

#include "rendering.cuh"

__device__
void renderer::ray_bounce(const uint t_in_block_idx, ray& r, const uint bounce_limit, curandState* const local_rand_state) {
	extern __shared__ char array[];

	//hit_record* hit_rec = &shared_hit_records[t_in_block_idx];

	//access shared memeory portion representing hit_records array
	uint block_size = blockDim.x * blockDim.y;
	hit_record* volatile hit_rec = (hit_record*)(&array[BVH_NODE_CACHE_SIZE * sizeof(bvh_node) + t_in_block_idx * sizeof(hit_record)]);
	//hit_record hit_rec;

	for (int n_bounces = 0; n_bounces < bounce_limit; n_bounces++) {

		if (!bvh::hit(r, 0.0f, FLT_MAX, *hit_rec, (bvh_node*)(&array[0]))) {
			//access portion of shared memory representing background
			r.mul_spectrum((float*)(&array[BVH_NODE_CACHE_SIZE * sizeof(bvh_node) + (block_size * sizeof(hit_record))]), N_CIE_SAMPLES);
			return;
		}

		if (!hit_rec->mat->scatter(r, *hit_rec, local_rand_state)) {
			return;
		}

	}

	//for (int i = 0; i < N_RAY_WAVELENGTHS; i++) {
	//	r.power_distr[i] = 0.0f;
	//}
	r.valid_wavelengths = 0;

}

__device__
point3 renderer::defocus_disk_sample(vec3 camera_center, vec3 defocus_disk_u, vec3 defocus_disk_v, curandState* local_rand_state) {
	// Returns a random point in the camera defocus disk.
	auto p = random_in_unit_disk(local_rand_state);
	return camera_center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
}

__device__
vec3 renderer::pixel_sample_square(const vec3 pixel_delta_u, const vec3 pixel_delta_v, curandState* local_rand_state) {
	// Returns a random point in the square surrounding a pixel at the origin.

	auto px = -0.5f + cuda_random_float(local_rand_state);
	auto py = -0.5f + cuda_random_float(local_rand_state);
	return (px * pixel_delta_u) + (py * pixel_delta_v);
};

__device__
vec3 renderer::pixel_stratified_sample_square(const uint sample_x, const uint sample_y, const float recip_sqrt_spp, const vec3 pixel_delta_u, const vec3 pixel_delta_v, curandState* local_rand_state) {
	// Returns a random point in the square surrounding a pixel at the origin.
	float px = -0.5f + recip_sqrt_spp * (float(sample_x) + cuda_random_float(local_rand_state));
	float py = -0.5f + recip_sqrt_spp * (float(sample_y) + cuda_random_float(local_rand_state));
	return (px * pixel_delta_u) + (py * pixel_delta_v);
};

__device__
ray renderer::get_ray(uint i, uint j, const point3 pixel00_loc, const vec3 pixel_delta_u, const vec3 pixel_delta_v,
	const point3 camera_center, const vec3 defocus_disk_u, const vec3 defocus_disk_v, const float defocus_angle,
	curandState* local_rand_state) {
	/*
	 * Get a randomly sampled camera ray for the pixel at location i,j
	 * originating from  a random point on the camera defocus disk
	 * NOTE: ray direction is not a unit vector in order to have a simpler and slightly faster code
	 */

	auto pixel_center = pixel00_loc + ((float)i * pixel_delta_u) + ((float)j * pixel_delta_v);
	auto pixel_sample = pixel_center + pixel_sample_square(pixel_delta_u, pixel_delta_v, local_rand_state);


	auto ray_origin = (defocus_angle <= 0.0f) ? camera_center : defocus_disk_sample(camera_center,
		defocus_disk_u,
		defocus_disk_v,
		local_rand_state);
	auto ray_direction = pixel_sample - ray_origin;

	return ray(ray_origin, ray_direction, local_rand_state);
};

__device__
ray renderer::get_ray_stratified_sample(uint i, uint j,
	const point3 pixel00_loc,
	const vec3 pixel_delta_u,
	const vec3 pixel_delta_v,
	const uint sample_x,
	const uint sample_y,
	const float recip_sqrt_spp,
	const point3 camera_center,
	const float defocus_angle,
	const vec3 defocus_disk_u,
	const vec3 defocus_disk_v,
	curandState* local_rand_state) {
	/*
	 * Get a randomly sampled camera ray for the pixel at location i,j
	 * originating from  a random point on the camera defocus disk
	 * NOTE: ray direction is not a unit vector in order to have a simpler and slightly faster code
	 */

	auto pixel_center = pixel00_loc + ((float)i * pixel_delta_u) + ((float)j * pixel_delta_v);
	auto pixel_sample = pixel_center + pixel_stratified_sample_square(sample_x, sample_y, recip_sqrt_spp, pixel_delta_u, pixel_delta_v, local_rand_state);

	auto ray_origin = (defocus_angle <= 0.0f) ? camera_center : defocus_disk_sample(camera_center,
		defocus_disk_u,
		defocus_disk_v,
		local_rand_state);
	auto ray_direction = pixel_sample - ray_origin;

	return ray(ray_origin, ray_direction, local_rand_state);
};

__global__
void init_random_states(int max_x, int max_y, curandState* rand_state) {
	//uint i = threadIdx.x + blockIdx.x * blockDim.x;
	//uint j = threadIdx.y + blockIdx.y * blockDim.y;

	uint block_size = blockDim.x * blockDim.y;
	uint block_idx = blockIdx.y * gridDim.x + blockIdx.x;
	uint thread_in_block_idx = threadIdx.y * blockDim.x + threadIdx.x;

	uint coalesced_global_idx = thread_in_block_idx + block_size * block_idx;
	//if ((i >= max_x) || (j >= max_y))
	//	return;


	//uint thread_index = j * max_x + i;

	//Each thread gets different seed, same sequence number, no offset
	curand_init(1984 + coalesced_global_idx, 0, 0, &rand_state[coalesced_global_idx]);
}

__device__
void save_to_fb(color& pixel_color, const uint coalesced_global_idx, const uint samples_per_pixel, float* fb_r, float* fb_g, float* fb_b) {

	pixel_color = expand_sRGB(XYZ_to_sRGB(pixel_color / float(samples_per_pixel), reinterpret_cast<const float*>(dev_d65_XYZ_to_sRGB)));

	//write color in coalesced fashion
	fb_r[coalesced_global_idx] = pixel_color[0];
	fb_g[coalesced_global_idx] = pixel_color[1];
	fb_b[coalesced_global_idx] = pixel_color[2];
}

__global__
void
spectral_render_kernel(float* fb_r, float* fb_g, float* fb_b, bvh** bvh, uint width, uint height, uint offset_x, uint offset_y, camera_data cam_data, float* background_spectrum,
	const short_uint samples_per_pixel, const short_uint bounce_limit, curandState* rand_state) {

	uint i = threadIdx.x + blockIdx.x * blockDim.x; //col idx
	uint j = threadIdx.y + blockIdx.y * blockDim.y; //row idx

	uint pixel_index = j * width + i;

	uint block_size = blockDim.x * blockDim.y;
	uint block_idx = blockIdx.y * gridDim.x + blockIdx.x;

	uint thread_in_block_idx = threadIdx.y * blockDim.x + threadIdx.x;
	uint coalesced_global_idx = thread_in_block_idx + block_size * block_idx;

	//INITIALIZE SHARED MEMORY HERE IF NEEDED

	//ray* shared_rays = (ray*)(&array[BVH_NODE_CACHE_SIZE * sizeof(bvh_node) + (block_size * sizeof(hit_record))]);
	if (thread_in_block_idx == 0) {
		extern __shared__ char array[];

		bvh_node* bvh_node_cache = (bvh_node*)(&array[0]);
		//access shared memory portion representing background spectrum
		float* sh_background_spectrum = (float*)(&array[BVH_NODE_CACHE_SIZE * sizeof(bvh_node) + (block_size * sizeof(hit_record)) /* + (block_size * sizeof(ray))*/]);

		//write values from global memory
		for (int k = 0; k < N_CIE_SAMPLES; k++) {
			sh_background_spectrum[k] = background_spectrum[k];
		}

		//move higher level nodes to shared memory
		if ((*bvh)->is_valid()) {
			(*bvh)->to_shared(bvh_node_cache, BVH_NODE_CACHE_SIZE);
		}
	}


	__syncthreads();

	if (i >= width || j >= height)
		return;

	//coalesced (as much as possible, since curandState size is > 4 bytes) read from rand state array
	curandState local_rand_state = rand_state[coalesced_global_idx];


	color pixel_color;

	if ((*bvh)->is_valid()) {
		for (short_uint k = 0; k < samples_per_pixel; k++) {
			/*
			* trace the ray from camera center to current pixel sample
			* then sum the sample color obtained from the spectrum to current pixel
			*/

			ray r = renderer::get_ray(offset_x + i, offset_y + j, cam_data.pixel00_loc, cam_data.pixel_delta_u, cam_data.pixel_delta_v,
				cam_data.camera_center, cam_data.defocus_disk_u, cam_data.defocus_disk_v,
				cam_data.defocus_angle, &local_rand_state);

			renderer::ray_bounce(thread_in_block_idx, r, bounce_limit, &local_rand_state);

			pixel_color += dev_spectrum_to_XYZ(r.wavelengths, r.power_distr, N_RAY_WAVELENGTHS, r.valid_wavelengths);
		}
	}

	//coalesced (as much as possible, since curandState size is > 4 bytes) write in rand state array for future use
	rand_state[coalesced_global_idx] = local_rand_state;

	save_to_fb(pixel_color, coalesced_global_idx, samples_per_pixel, fb_r, fb_g, fb_b);
}

__host__
void renderer::assign_cam_data(camera* cam) {
	cam_data = camera_data(cam->getImageWidth(), cam->getImageHeight(), cam->getPixelDeltaU(), cam->getPixelDeltaV(),
		cam->getPixel00Loc(), cam->getDefocusAngle(), cam->getCenter(),
		cam->getDefocusDiskU(), cam->getDefocusDiskV());
}

__host__
void renderer::call_render_kernel(short_uint width, short_uint height, short_uint offset_x, short_uint offset_y) {

	if (!device_inited) {
		cerr << "Device parameters were not initialized, render aborted" << endl;
		return;
	}

	// clock_t start, stop;
	//start = clock();

	/*
	camera_data cam_data = camera_data(cam->getImageWidth(), cam->getImageHeight(), cam->getPixelDeltaU(), cam->getPixelDeltaV(),
									   cam->getPixel00Loc(), cam->getDefocusAngle(), cam->getCenter(),
									   cam->getDefocusDiskU(), cam->getDefocusDiskV());*/

	spectral_render_kernel << <blocks, threads, shared_mem_size >> > (dev_fb_r, dev_fb_g, dev_fb_b,
		dev_bvh,
		width,
		height,
		offset_x,
		offset_y,
		cam_data,
		dev_background_spectrum,
		samples_per_pixel,
		bounce_limit,
		dev_rand_state);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
};

__host__
void renderer::init_device_params(const dim3 _threads, const dim3 _blocks, const uint _max_chunk_width, const uint _max_chunk_height) {
	// Allocate Frame Buffer
	//vec3* dev_fb = nullptr;
	threads = _threads;
	blocks = _blocks;
	max_chunk_width = _max_chunk_width;
	max_chunk_height = _max_chunk_height;

	//    PREPARE SHARED MEMORY SIZE HERE

	uint shared_bg_size = N_CIE_SAMPLES * sizeof(float);
	cout << "shared_bg_size is " << shared_bg_size << endl;
	uint node_cache_size = BVH_NODE_CACHE_SIZE * sizeof(bvh_node);
	cout << "node_cache_size is " << node_cache_size << endl;
	uint shared_hit_rec_size = threads.x * threads.y * sizeof(hit_record);
	cout << "shared_hit_rec_size is " << shared_hit_rec_size << endl;
	//uint shared_rays_size = threads.x * threads.y * sizeof(ray);
	//cout << "shared_rays_size is " << shared_rays_size << endl;

	shared_mem_size = shared_bg_size + node_cache_size + shared_hit_rec_size /*+ shared_rays_size*/;
	uint max_num_pixels = max_chunk_width * max_chunk_height;
	//allocate red buffer
	checkCudaErrors(cudaMalloc((void**)&dev_fb_r, /*max_num_pixels*/ threads.x * blocks.x * threads.y * blocks.y * sizeof(float)));
	checkCudaErrors(cudaGetLastError());

	//allocate green buffer
	checkCudaErrors(cudaMalloc((void**)&dev_fb_g,  /*max_num_pixels*/ threads.x * blocks.x * threads.y * blocks.y * sizeof(float)));
	checkCudaErrors(cudaGetLastError());

	//allocate blue buffer
	checkCudaErrors(cudaMalloc((void**)&dev_fb_b,  /*max_num_pixels*/ threads.x * blocks.x * threads.y * blocks.y * sizeof(float)));
	checkCudaErrors(cudaGetLastError());

	//allocate random state
	checkCudaErrors(cudaMalloc((void**)&dev_rand_state, /*max_num_pixels */ threads.x * blocks.x * threads.y * blocks.y * sizeof(curandState)));
	checkCudaErrors(cudaGetLastError());

	//allocate and initialize background spectrum
	float h_background_spectrum[N_CIE_SAMPLES];
	checkCudaErrors(cudaMalloc((void**)&dev_background_spectrum, N_CIE_SAMPLES * sizeof(float)));
	checkCudaErrors(cudaGetLastError());

	srgb_to_illuminance_spectrum(background, h_background_spectrum);
	checkCudaErrors(cudaMemcpy(dev_background_spectrum, h_background_spectrum, N_CIE_SAMPLES * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//initialize random state
	init_random_states << <blocks, threads >> > (max_chunk_width, max_chunk_height, dev_rand_state);

	//initialize intersection data buffer
	//alloc_intersection_data << <1, 1 >> > (inters_data_buffer, n_streams, _threads.x * _threads.y, _blocks.x * _blocks.y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	auto lc = log_context::getInstance();
	lc->add_entry("chunk width", max_chunk_width);
	lc->add_entry("chunk height", max_chunk_height);

	lc->add_entry("chunk total byte size", max_num_pixels * sizeof(vec3));
	lc->add_entry("shared memory byte size", shared_mem_size);
	lc->add_entry("threads x", threads.x);
	lc->add_entry("threads y", threads.y);
	lc->add_entry("threads z", threads.z);

	lc->add_entry("blocks x", blocks.x);
	lc->add_entry("blocks y", blocks.y);
	lc->add_entry("blocks z", blocks.z);

	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, spectral_render_kernel);
	cout << "Max threads per block: " << attr.maxThreadsPerBlock << endl;
	cout << "Registers per thread: " << attr.numRegs << endl;

	device_inited = true;
}

__host__
void renderer::clean_device() {
	if (device_inited) {
		//dealloc_intersection_data << <1, 1 >> > (inters_data_buffer, n_streams);
		//checkCudaErrors(cudaFree(inters_data_buffer));
		checkCudaErrors(cudaFree(dev_rand_state));
		checkCudaErrors(cudaFree(dev_fb_r));
		checkCudaErrors(cudaFree(dev_fb_b));
		checkCudaErrors(cudaFree(dev_fb_g));
		checkCudaErrors(cudaFree(dev_background_spectrum));
		checkCudaErrors(cudaGetLastError());
	}
}
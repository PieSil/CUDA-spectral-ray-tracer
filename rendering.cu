//
// Created by pietr on 09/04/2024.
//

#include "rendering.cuh"


//TODO: fix comments

__device__
void ray_bounce(ray &r, const float *background_emittance_spectrum, bvh* bvh, const uint bounce_limit, curandState *local_rand_state) {

    //TODO: finish this

    hit_record rec;
    //ray cur_ray = r;

    for (int n_bounces = 0; n_bounces < bounce_limit; n_bounces++) {

        if (!bvh->hit(r, 0.0f, FLT_MAX, rec)) {
            //float random = cuda_random_float(local_rand_state);

            for (int i = 0; i < N_RAY_WAVELENGTHS; i++) {
                float lambda = r.wavelengths[i];
                float weight = spectrum_interp(background_emittance_spectrum, lambda);
                r.power_distr[i] *= weight;
            }

            return; //background * attenuation;
        }

        if (!rec.mat->scatter(r, rec, local_rand_state)) {
            return;
        }
    }

    for(int i = 0; i < N_RAY_WAVELENGTHS; i++) {
        r.power_distr[i] = 0.0f;
    }


    //Max bounces reached, return black
    //return attenuation;
    //return color(0,0,0);
    //return ambient_color + final_color;
}

__device__
point3 defocus_disk_sample(vec3 camera_center, vec3 defocus_disk_u, vec3 defocus_disk_v, curandState* local_rand_state) {
    // Returns a random point in the camera defocus disk.
    auto p = random_in_unit_disk(local_rand_state);
    return camera_center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
}

__device__
vec3 pixel_sample_square(const vec3 pixel_delta_u, const vec3 pixel_delta_v, curandState* local_rand_state) {
    // Returns a random point in the square surrounding a pixel at the origin.

    auto px = -0.5f + cuda_random_float(local_rand_state);
    auto py = -0.5f + cuda_random_float(local_rand_state);
    return (px * pixel_delta_u) + (py * pixel_delta_v);
};

__device__
vec3 pixel_stratified_sample_square(const uint sample_x, const uint sample_y, const float recip_sqrt_spp, const vec3 pixel_delta_u, const vec3 pixel_delta_v, curandState* local_rand_state) {
    // Returns a random point in the square surrounding a pixel at the origin.
    float px = -0.5f + recip_sqrt_spp * (float(sample_x) + cuda_random_float(local_rand_state));
    float py = -0.5f + recip_sqrt_spp * (float(sample_y) + cuda_random_float(local_rand_state));
    return (px * pixel_delta_u) + (py * pixel_delta_v);
};

__device__
ray get_ray(uint i, uint j, const point3 pixel00_loc, const vec3 pixel_delta_u, const vec3 pixel_delta_v,
            const point3 camera_center, const vec3 defocus_disk_u, const vec3 defocus_disk_v, const float defocus_angle,
            curandState *local_rand_state) {
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
ray get_ray_stratified_sample(uint i, uint j,
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
void render_init(int max_x, int max_y, curandState *rand_state) {
    uint i = threadIdx.x + blockIdx.x * blockDim.x;
    uint j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= max_x) || (j >= max_y))
        return;

    uint thread_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    //curand_init(1984, pixel_index, 0, &rand_state[thread_index]);

    //Each thread gets different seed, same sequence number, no offset
    curand_init(1999+thread_index, 0, 0, &rand_state[thread_index]);
}

__device__
void save_to_fb(color pixel_color, uint pixel_index, uint samples_per_pixel, vec3* fb) {
    fb[pixel_index] = expand_sRGB(XYZ_to_sRGB(pixel_color/float(samples_per_pixel), reinterpret_cast<const float *>(dev_d65_sRGB_to_XYZ)));
    //fb[pixel_index] = pixel_color/float(samples_per_pixel);
}

__global__
void spectral_render_kernel(vec3 *fb, bvh* bvh, const uint samples_per_pixel, const uint max_x, const uint max_y,
                                   const uint bounce_limit, const vec3 pixel_delta_u, const vec3 pixel_delta_v,
                                   const point3 pixel00_loc, const float defocus_angle, const point3 camera_center,
                                   const vec3 defocus_disk_v, const float *background_spectrum, curandState *rand_state,
                                   const vec3 defocus_disk_u) {
    uint i = threadIdx.x + blockIdx.x * blockDim.x;
    uint j = threadIdx.y + blockIdx.y * blockDim.y;

    extern __shared__ char array[];

    if((i >= max_x) || j >= max_y)
        return;

    uint pixel_index = j*max_x + i;

    //INITIALIZE SHARED MEMORY HERE IF NEEDED
    uint thread_in_block_idx = threadIdx.x*blockDim.y + threadIdx.y;
    float* sh_background_spectrum = (float*)array;
    if (thread_in_block_idx == 0) {
        for(int k = 0; k < N_CIE_SAMPLES; k++) {
            sh_background_spectrum[k] = background_spectrum[k];
        }
    }
//    uint thread_in_block_idx = threadIdx.x*blockDim.y + threadIdx.y;
//    int *light_indices = (int *) array;

//    if (thread_in_block_idx == 0) {
//        int shadow_ray_iterations = int(min(l_list->n_lights, l_list->max_rays));
//
//        for (int offset = 0; offset < blockDim.x * blockDim.y * shadow_ray_iterations; offset += shadow_ray_iterations) {
//            for (int k = 0; k < shadow_ray_iterations; k++) {
//                light_indices[offset + k] = k;
//            }
//        }
//
//        /*
//         * DEBUG PURPOSES
//        if (pixel_index == 0) {
//            for (int offset = 0; offset < blockDim.x * blockDim.y * shadow_ray_iterations; offset++) {
//                printf("light indices [%d] = %d\n", offset, light_indices[offset]);
//            }
//        }
//         */
//    }

    __syncthreads();

    curandState local_rand_state = rand_state[pixel_index];
    color pixel_color;

    //TODO: one thread per sample?
    for (int k = 0; k < samples_per_pixel; k++) {
        /*
        * trace the ray from camera center to current pixel sample
        * then sum the sample color obtained from the spectrum to current pixel
        */
        ray r = get_ray(i, j, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center, defocus_disk_u,
                        defocus_disk_v, defocus_angle, &local_rand_state);
        ray_bounce(r, sh_background_spectrum, bvh, bounce_limit, &local_rand_state);
        pixel_color += dev_spectrum_to_XYZ(r.wavelengths, r.power_distr, N_RAY_WAVELENGTHS);
    }

    //save updated rand state in random state array for future use
    rand_state[pixel_index] = local_rand_state;


    save_to_fb(pixel_color, pixel_index, samples_per_pixel, fb);
    //save_to_fb(pixel_color, pixel_index, samples_per_pixel, fb);
}

__host__
void call_render_kernel(bvh *bvh, uint samples_per_pixel, const camera *cam, uint bounce_limit, dim3 blocks,
                        dim3 threads) {

    int image_width = cam->getImageWidth();
    int image_height = cam->getImageHeight();

    // Define Frame Buffer size
    uint num_pixels = cam->getNumPixels();
    size_t fb_size = num_pixels*sizeof(vec3);

    // Allocate Frame Buffer
    // TODO: use Unified Memory instead?
    vec3* fb = new vec3[fb_size];
    vec3* dev_fb = nullptr;
    checkCudaErrors(cudaMalloc((void**) &dev_fb, fb_size));

    curandState *dev_rand_state;
    checkCudaErrors(cudaMalloc((void**)&dev_rand_state, num_pixels*sizeof(curandState)));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    float h_background_spectrum[N_CIE_SAMPLES];
    float* dev_background_spectrum;
    checkCudaErrors(cudaMalloc((void**)&dev_background_spectrum, N_CIE_SAMPLES*sizeof(float)));
    srgb_to_spectrum(cam->getBackground(), h_background_spectrum);
    checkCudaErrors(cudaMemcpy(dev_background_spectrum, h_background_spectrum, N_CIE_SAMPLES * sizeof(float), cudaMemcpyHostToDevice));


    render_init<<<blocks, threads>>>(cam->getImageWidth(), cam->getImageHeight(), dev_rand_state);

//    PREPARE SHARED MEMORY SIZE HERE IF NEEDED
//    uint* h_n_lights = new uint;
//    uint* h_max_rays = new uint;
//    checkCudaErrors(cudaMemcpy(h_n_lights, &(l_list->n_lights), sizeof(uint), cudaMemcpyDeviceToHost));
//    checkCudaErrors(cudaMemcpy(h_max_rays, &(l_list->max_rays), sizeof(uint), cudaMemcpyDeviceToHost));
//    int shadow_ray_iterations = int(min(*h_n_lights, *h_max_rays));
//    free(h_n_lights);
//    free(h_max_rays);

//    unsigned shared_mem_size = (shadow_ray_iterations*sizeof(int)*threads.x*threads.y*threads.z);

    unsigned shared_mem_size = (N_CIE_SAMPLES);

    clock_t start, stop;
    start = clock();

    spectral_render_kernel<<<blocks, threads, shared_mem_size>>>(dev_fb,
                                                                 bvh,
                                                                 samples_per_pixel,
                                                                 cam->getImageWidth(),
                                                                 cam->getImageHeight(),
                                                                 bounce_limit,
                                                                 cam->getPixelDeltaU(),
                                                                 cam->getPixelDeltaV(),
                                                                 cam->getPixel00Loc(),
                                                                 cam->getDefocusAngle(),
                                                                 cam->getCenter(),
                                                                 cam->getDefocusDiskV(),
                                                                 dev_background_spectrum,
                                                                 dev_rand_state,
                                                                 cam->getDefocusDiskU());


    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(dev_rand_state));

    checkCudaErrors(cudaMemcpy(fb, dev_fb, fb_size, cudaMemcpyDeviceToHost));

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::clog << "took " << timer_seconds << " seconds.\n";

    //free device memory
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(dev_fb));
    checkCudaErrors(cudaFree(dev_background_spectrum));

    write_to_ppm(fb, image_width, image_height);

    //free host memory
    free(fb);
};
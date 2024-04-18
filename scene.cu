//
// Created by pietr on 13/04/2024.
//

#include "scene.cuh"

using namespace scene;

__global__
void create_bvh_kernel(hittable **d_world, size_t world_size, bvh **d_bvh, bool *success) {

    curandState _rand_state = curandState();
    curandState* rand_state = &_rand_state;
    curand_init(1984, 0, 0, rand_state);

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_bvh = new bvh(d_world, world_size, rand_state);
        *success = (*d_bvh)->is_valid();
    }
}

__global__
void create_world_kernel(uint world_selector, hittable **d_list, material **d_mat_list, int *world_size, int *n_materials,
                    float* dev_sRGBToSpectrum_Data) {

    /*
     * initialize hittables and materials based on a world selector
     */

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        switch(world_selector){
            case 0:
                device_random_world(d_list, d_mat_list, world_size, n_materials);
                break;
            case 1:
                device_quad_world(d_list, d_mat_list);
                break;

            case 2:
                device_simple_light(d_list, d_mat_list);
                break;
            case 3:
                device_cornell_box(d_list, d_mat_list);
                break;
            default:
                //device_simple_light(d_list, d_mat_list);
                device_random_world(d_list, d_mat_list, world_size, n_materials);
        }

        /*
         * precompute reflectance and emittance spectrum
         */

        
        for(int i = 0; i < *n_materials; i++) {
            d_mat_list[i]->compute_albedo_spectrum(dev_sRGBToSpectrum_Data);
            d_mat_list[i]->compute_emittance_spectrum(dev_sRGBToSpectrum_Data);
        }
        

    }
}

__global__ void
free_world_kernel(hittable **d_list, material **d_mat_list, int world_size, int n_materials, bvh **dev_bvh) {

    if (d_list != nullptr && threadIdx.x == 0 && blockIdx.x == 0) {

        for (int i = 0; i < n_materials; i++) {
            delete *(d_mat_list+i);
        }

        for (int i = 0; i < world_size; i++) {
            delete *(d_list+i);
        }

        delete *dev_bvh;
    }
}

__device__
void scene::device_random_world(hittable **d_list, material **d_mat_list, int *world_size, int *n_materials) {
    curandState _rand_state = curandState();
    curandState* rand_state = &_rand_state;
    curand_init(1999, 0, 0, rand_state);
    int void_positions = 0;

    for (int a = -HALF_N_RANDOM_I; a < HALF_N_RANDOM_I; a++) {
        for (int b = -HALF_N_RANDOM_J; b < HALF_N_RANDOM_J; b++) {
            auto choose_mat = cuda_random_float(rand_state);
            point3 center(float(a) + 0.9f*cuda_random_float(rand_state), 0.2f, float(b) + 0.9f*cuda_random_float(rand_state));
            int list_idx = (a+HALF_N_RANDOM_I)*2*HALF_N_RANDOM_J + (b+HALF_N_RANDOM_J)-void_positions;

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {

                if (choose_mat < 0.8f) {
                    // diffuse
                    auto albedo = color::random(rand_state) * color::random(rand_state);
                    *(d_mat_list+list_idx) = new material();
                    **(d_mat_list+list_idx) = material::lambertian(albedo);
                    *(d_list+list_idx) = new sphere(center, 0.2f, *(d_mat_list+list_idx));
                } else if (choose_mat < 0.95f) {
                    // metal
                    //auto albedo = color::random(rand_state);
                    auto albedo = color::random(0.5f, 1.0f, rand_state);
                    auto fuzz = cuda_random_float(0.0f, 0.5f, rand_state);
                    *(d_mat_list+list_idx) = new material();
                    **(d_mat_list+list_idx) = material::metallic(albedo, fuzz);
                    *(d_list+list_idx) = new sphere(center, 0.2f, *(d_mat_list+list_idx));
                } else {
                    // glass
                    *(d_mat_list+list_idx) = new material();
                    **(d_mat_list+list_idx) = material::dielectric(1.5);
                    *(d_list+list_idx) = new sphere(center, 0.2f, *(d_mat_list+list_idx));
                }

                /*if (*(d_list+list_idx) == nullptr)
                    printf("list[%d] is nullptr\n");
                if (*(d_mat_list+list_idx) == nullptr)
                    printf("material[%d] is nullptr\n");*/

            } else {
                void_positions++;
            }
        }
    }

    *(d_mat_list+N_RANDOM_MATERIALS-void_positions) = new material();
    **(d_mat_list+N_RANDOM_MATERIALS-void_positions) = material::lambertian(color(0.5, 0.5, 0.5));
    *(d_list+N_RANDOM_SPHERES-void_positions) = new sphere(point3(0,-1000,0), 1000, *(d_mat_list+N_RANDOM_MATERIALS-void_positions));

    *(d_mat_list+N_RANDOM_MATERIALS-void_positions+1) = new material();
    **(d_mat_list+N_RANDOM_MATERIALS-void_positions+1) = material::dielectric(1.5f);
    *(d_list+N_RANDOM_SPHERES-void_positions+1) = new sphere(point3(0, 1, 0), 1.0, *(d_mat_list+N_RANDOM_MATERIALS-void_positions+1));

    *(d_mat_list+N_RANDOM_MATERIALS-void_positions+2) = new material();
    **(d_mat_list+N_RANDOM_MATERIALS-void_positions+2) = material::lambertian(color(0.4, 0.2, 0.1));
    *(d_list+N_RANDOM_SPHERES-void_positions+2) = new sphere(point3(-4, 1, 0), 1.0, *(d_mat_list+N_RANDOM_MATERIALS-void_positions+2));

    *(d_mat_list+N_RANDOM_MATERIALS-void_positions+3) = new material();
    **(d_mat_list+N_RANDOM_MATERIALS-void_positions+3) = material::metallic(color(0.7, 0.6, 0.5), 0.0);
    *(d_list+N_RANDOM_SPHERES-void_positions+3) = new sphere(point3(4, 1, 0), 1.0, *(d_mat_list+N_RANDOM_MATERIALS-void_positions+3));

    if (world_size != nullptr)
        *world_size = *world_size-void_positions;

    if (n_materials != nullptr)
        *n_materials = *n_materials-void_positions;

    delete rand_state;
}

__device__
void scene::device_quad_world(hittable **d_list, material **d_mat_list) {
    d_mat_list[0] = new material();
    *d_mat_list[0] = material::lambertian(color(1.0, 0.2, 0.2));
    d_mat_list[1] = new material();
    *d_mat_list[1] = material::lambertian(color(0.2, 1.0, 0.2));
    d_mat_list[2] = new material();
    *d_mat_list[2] = material::lambertian(color(0.2, 0.2, 1.0));
    d_mat_list[3] = new material();
    *d_mat_list[3] = material::lambertian(color(1.0, 0.5, 0.0));
    d_mat_list[4] = new material();
    *d_mat_list[4]= material::lambertian(color(0.2, 0.8, 0.8));

    d_list[0] = new quad(point3(-3,-2, 5), vec3(0, 0,-4), vec3(0, 4, 0), d_mat_list[0]);
    d_list[1] = new quad(point3(-2,-2, 0), vec3(4, 0,0), vec3(0, 4, 0), d_mat_list[1]);
    d_list[2] = new quad(point3(3,-2, 1), vec3(0, 0,4), vec3(0, 4, 0), d_mat_list[2]);
    d_list[3] = new quad(point3(-2,3, 1), vec3(4, 0, 0), vec3(0, 0, 4), d_mat_list[3]);
    d_list[4] = new quad(point3(-2,-3, 5), vec3(4, 0, 0), vec3(0, 0, -4), d_mat_list[4]);
}

__device__
void scene::device_simple_light(hittable **d_list, material **d_mat_list) {

    d_mat_list[0] = new material();
    *(d_mat_list[0]) = material::lambertian(color(.0f, 1.0f, .0f));

    d_mat_list[1] = new material();
    //*(d_mat_list[1]) = material::metallic(color(.5f, .5f, .5f), .5f);
    //*(d_mat_list[1]) = material::dielectric(1.5f);
    *(d_mat_list[1]) = material::lambertian(color(.1f, .5f, .7f));

    d_mat_list[2] = new material();
    *(d_mat_list[2]) = material::emissive(color(1.0f, 1.0f, 1.0f), 10.0f);

    d_list[0] = new sphere(point3(0, -1000, 0), 1000, d_mat_list[0]);
    d_list[1] = new sphere(point3(0, 2, 0), 2, d_mat_list[1]);
    d_list[2] = new quad(point3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), d_mat_list[2]);
}

__device__
void scene::device_cornell_box(hittable **d_list, material **d_mat_list) {
    d_mat_list[0] = new material();
    *d_mat_list[0] = material::lambertian(color(.65, .05, .05));
    d_mat_list[1] = new material();
    *d_mat_list[1] = material::lambertian(color(.73, .73, .73));
    d_mat_list[2] = new material();
    *d_mat_list[2] = material::lambertian(color(.12, .45, .15));
    d_mat_list[3] = new material();
    *d_mat_list[3] = material::emissive(color(15, 15, 15));

//    d_list[0] = new quad(point3(555, 0, 0), vec3(0, 555, 0), vec3 (0, 0, 555), d_mat_list[2]);
//    d_list[1] = new quad(point3(0, 0, 0), vec3(0, 555, 0), vec3 (0, 0, 555), d_mat_list[0]);
//    d_list[2] = new quad(point3(343, 554, 332), vec3(-130, 0, 0), vec3 (0, 0, -105), d_mat_list[3]);
//    d_list[3] = new quad(point3(0, 0, 0), vec3(555, 0, 0), vec3 (0, 0, 555), d_mat_list[1]);
//    d_list[4] = new quad(point3(555, 555, 555), vec3(-555, 0, 0), vec3 (0, 0, -555), d_mat_list[1]);
//    d_list[5] = new quad(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), d_mat_list[1]);

}

__host__
void scene::init_world_parameters(uint world_selector, int *world_size_ptr, int *n_materials_ptr) {

    /*
     * select correct parameters (hard-coded) based on world selector
     */

    switch (world_selector) {
        case 0:
            //random world
            *world_size_ptr = RANDOM_WORLD_SIZE;
            *n_materials_ptr = RANDOM_WORLD_MATERIALS;
            break;

        case 1:
            //quads
            *world_size_ptr = 5;
            *n_materials_ptr = 5;
            break;

        case 2:
            *world_size_ptr = 3;
            *n_materials_ptr = 3;
            break;

        case 3:
            *world_size_ptr = 6;
            *n_materials_ptr = 4;
            break;

        default:
            //random world
            *world_size_ptr = RANDOM_WORLD_SIZE;
            *n_materials_ptr = RANDOM_WORLD_MATERIALS;
            break;
    }
}

__host__
camera_builder scene::random_world_cam_builder() {
    float vfov = 20.0f;
    point3 lookfrom = point3(13,2,3);
    point3 lookat = point3(0,0,0);
    vec3 vup = vec3(0,1,0);
    float defocus_angle = 0.6f;
    float focus_dist = 10.0f;
    color background = color(0.70, 0.80, 1.00);

    return camera_builder().
            setAspectRatio(16.0f/9.0f).
            setImageWidth(400).
            setVfov(vfov).
            setLookfrom(lookfrom).
            setVup(vup).
            setLookat(lookat).
            setDefocusAngle(defocus_angle).
            setFocusDist(focus_dist).
            setBackground(background);
}

__host__
camera_builder scene::quad_world_camera_builder() {
    float vfov = 80.0f;
    point3 lookfrom = point3(0,0,9);
    point3 lookat = point3(0,0,0);
    vec3 vup = vec3(0,1,0);
    float defocus_angle = 0.0f;
    float focus_dist = 10.0f;
    color background = color(0.70, 0.80, 1.00);

    return camera_builder().
            setAspectRatio(1.0f).
            setImageWidth(400).
            setVfov(vfov).
            setLookfrom(lookfrom).
            setVup(vup).
            setLookat(lookat).
            setDefocusAngle(defocus_angle).
            setFocusDist(focus_dist).
            setBackground(background);
}

__host__
camera_builder scene::simple_light_camera_builder() {
    float vfov = 20.0f;
    point3 lookfrom = point3(26,3,6);
    //point3 lookfrom = point3(3,26,6);
    point3 lookat = point3(0,2,0);
    vec3 vup = vec3(0,1,0);
    float defocus_angle = 0.0f;
    float focus_dist = 10.0f;
    //color expanded_b_color = color(10.0f, 180.0f, 186.0f);
    //color expanded_b_color = color(255.0f, 255.0f, 255.0f);
    //color background = expanded_b_color/255;
    //color background = color(0.5f, 0.5f, 0.5f);
    color background = color(0.0f, 0.0f, 0.0f);
    //color background = color(1.00, 1.00, 1.00);

    return camera_builder().
            setAspectRatio(16.0f/9.0f).
            setImageWidth(400).
            setVfov(vfov).
            setLookfrom(lookfrom).
            setVup(vup).
            setLookat(lookat).
            setDefocusAngle(defocus_angle).
            setFocusDist(focus_dist).
            setBackground(background);
}

__host__
camera_builder scene::cornell_box_camera_builder() {
    float vfov = 40.0f;
    point3 lookfrom = point3(278,278,-800);
    point3 lookat = point3(278,278,0);
    vec3 vup = vec3(0,1,0);
    float defocus_angle = 0.0f;
    float focus_dist = 10.0f;
    color background = color(0.0f, 0.0f, 0.0f);
    //color background = color(0.70, 0.80, 1.00);

    return camera_builder().
            setAspectRatio(1.0f).
            setImageWidth(600).
            setVfov(vfov).
            setLookfrom(lookfrom).
            setVup(vup).
            setLookat(lookat).
            setDefocusAngle(defocus_angle).
            setFocusDist(focus_dist).
            setBackground(background);
}

__host__
bool scene::create_bvh(hittable** d_world, size_t world_size, bvh** d_bvh) {
    bool h_success;
    bool* d_success;

    checkCudaErrors(cudaMalloc((void**)&d_success, sizeof(bool)));
    create_bvh_kernel<<<1, 1>>>(d_world, world_size, d_bvh, d_success);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(&h_success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_success));
    checkCudaErrors(cudaDeviceSynchronize());

    return h_success;
}

__host__
void scene::create_world(hittable **d_list, material **d_mat_list, int *world_size, int *n_materials,
                  float* dev_sRGBToSpectrum_Data) {
    create_world_kernel<<<1, 1>>>(WORLD_SELECTOR, d_list, d_mat_list, world_size, n_materials, dev_sRGBToSpectrum_Data);
}

__host__
void scene::free_world(hittable **d_list, bvh **dev_bvh, material **d_mat_list, int world_size,
                int n_materials) {
    free_world_kernel<<<1, 1>>>(d_list, d_mat_list, world_size, n_materials, dev_bvh);

}

const result scene_manager::init_world() {
        /*
     * Allocate memory on GPU for hittables and materials, initialize their contents based on a world selector
     * then build a BVH
     */

        int *dev_n_materials_ptr = nullptr;
        int *dev_world_size_ptr = nullptr;

        //select the correct values for size and #materials
        init_world_parameters(WORLD_SELECTOR, h_world_size_ptr, h_n_materials_ptr);

        //copy parameters to device memory so they can be modified from device code
        checkCudaErrors(cudaMalloc((void **) &dev_n_materials_ptr, sizeof(int)));
        checkCudaErrors(cudaMalloc((void **) &dev_world_size_ptr, sizeof(int)));
        checkCudaErrors(cudaMemcpy(dev_n_materials_ptr, h_n_materials_ptr, sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(dev_world_size_ptr, h_world_size_ptr, sizeof(int), cudaMemcpyHostToDevice));

        //allocate space for world
        checkCudaErrors(cudaMalloc((void **) &dev_world, *(h_world_size_ptr) * sizeof(hittable *)));
        if (*(h_world_size_ptr) > 0 && dev_world == nullptr) {
            return {false, "Not enough memory on device for dev_list\n"};
        }

        //allocate space for materials
        checkCudaErrors(cudaMalloc((void **) &dev_mat_list, (*h_n_materials_ptr) * sizeof(material *)));
        if ((*h_n_materials_ptr) > 0 && dev_mat_list == nullptr) {
            return {false, "Not enough memory on device for dev_mat_list\n"};
        }

        //allocate space for bvh
        checkCudaErrors(cudaMalloc((void **) &dev_bvh, sizeof(bvh *)));
        if (dev_bvh == nullptr) {
            return {false, "Not enough memory on device for BVH\n"};
        }

        //copy constant to device global memory (cannot use constant memory since table is too big)
        float *dev_ColorToSpectrum_Data;
        checkCudaErrors(cudaMalloc((void **) &dev_ColorToSpectrum_Data, 3 * 64 * 64 * 64 * 3 * sizeof(float)));
        if (dev_ColorToSpectrum_Data == nullptr) {
            return { false, "Not enough memory on device for dev_ColorToSpectrum_Data\n" };
        }
        checkCudaErrors(
                cudaMemcpy(dev_ColorToSpectrum_Data, sRGBToSpectrumTable_Data, 3 * 64 * 64 * 64 * 3 * sizeof(float),
                           cudaMemcpyHostToDevice));

        //build hittables and materials
        create_world(dev_world, dev_mat_list, dev_world_size_ptr, dev_n_materials_ptr, dev_ColorToSpectrum_Data);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        //cleanup
        checkCudaErrors(cudaFree(dev_ColorToSpectrum_Data));
        checkCudaErrors(cudaMemcpy(h_world_size_ptr, dev_world_size_ptr, sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_n_materials_ptr, dev_n_materials_ptr, sizeof(int), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(dev_world_size_ptr));
        checkCudaErrors(cudaFree(dev_n_materials_ptr));

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        //build bvh
        bool bvh_valid = create_bvh(dev_world, *h_world_size_ptr, dev_bvh);
        if (!bvh_valid && *h_world_size_ptr > 0) {
            return {false, "Error building BVH\n"};
        }

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        world_inited = true;
        return {true, "World created"};
}

__host__
void scene_manager::init_camera() {
    switch (WORLD_SELECTOR) {
        case 0:
            cam = random_world_cam_builder().getCamera();
            break;
        case 1:
            cam = quad_world_camera_builder().getCamera();
            break;
        case 2:
            cam = simple_light_camera_builder().getCamera();
            break;
        case 3:
            cam = cornell_box_camera_builder().getCamera();
            break;

        default:
            cam = random_world_cam_builder().getCamera();
            break;
    }

    cam_inited = true;
}

__host__
void scene_manager::destroy_world() {
    free_world(dev_world, dev_bvh, dev_mat_list, *h_world_size_ptr, *h_n_materials_ptr);
    checkCudaErrors(cudaFree(dev_world));
    checkCudaErrors(cudaFree(dev_bvh));
    checkCudaErrors(cudaFree(dev_mat_list));

    free(h_world_size_ptr);
    free(h_n_materials_ptr);
}
//
// Created by pietr on 13/04/2024.
//

#include "scene.cuh"

using namespace scene;

__global__
void create_bvh_kernel(tri** d_world, size_t world_size, bvh** d_bvh, bool* success) {

	curandState _rand_state = curandState();
	curandState* rand_state = &_rand_state;
	curand_init(1984, 0, 0, rand_state);

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*d_bvh = new bvh(d_world, world_size, rand_state);
		*success = (*d_bvh)->is_valid();
	}
}

__global__
void create_world_kernel(uint world_selector, tri** d_list, material* d_mat_list, int* world_size, int* n_materials, float* dev_sRGBToSpectrum_Data) {

	/*
	 * initialize hittables and materials based on a world selector
	 */

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		switch (world_selector) {

		case PRISM:
			device_prism_test(d_list, d_mat_list);
			break;

		case TRIS:
			device_different_mats_world(d_list, d_mat_list);
			break;

		case CORNELL:
		default:
			device_cornell_box(d_list, d_mat_list);
			break;
		}

		/*
		 * precompute reflectance and emittance spectrum
		 */
		for (int i = 0; i < *n_materials; i++) {
			d_mat_list[i].compute_spectral_distr(dev_sRGBToSpectrum_Data);
		}

	}
}

__global__ void
free_world_kernel(tri** d_list, int world_size, int n_materials, bvh** dev_bvh) {

	if (d_list != nullptr && threadIdx.x == 0 && blockIdx.x == 0) {

		//for (int i = 0; i < n_materials; i++) {
		//	delete* (d_mat_list + i);
		//}

		for (int i = 0; i < world_size; i++) {
			delete* (d_list + i);
		}

		delete* dev_bvh;
	}
}

__device__
void scene::device_cornell_box(tri** d_list, material* d_mat_list) {
	d_mat_list[0] =  material::lambertian(color(.65f, .05f, .05f)); //red
	d_mat_list[1] =  material::lambertian(color(.12f, .45f, .15f)); //green
	d_mat_list[2] =  material::dielectric(dev_flint_glass_b, dev_flint_glass_c);
	d_mat_list[3] = material::lambertian(color(.73f, .73f, .73f)); //white
	d_mat_list[4] = material::emissive(color(1.f, 1.f, 1.f), 5.f); //light
	d_mat_list[5] = material::metallic(color(.5f, .5f, .5f), 0.3f); //metal
	d_mat_list[6] = material::lambertian(color(.12f, .15f, .45f)); //blue

	tri** bottom_faces = &d_list[0];
	tri** top_faces = &d_list[2];
	tri** back_faces = &d_list[4];
	tri** left_faces = &d_list[6];
	tri** right_faces = &d_list[8];
	tri** light_faces = &d_list[10];

	//walls
	tri_quad bottom(point3(0, 0, 0), vec3(0, 0, 555), vec3(555, 0, 0), (const uint)3, bottom_faces);
	tri_quad back(point3(0, 0, 555.f), vec3(0, 555, 0), vec3(555, 0, 0), (const uint)3, back_faces);
	tri_quad top(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), (const uint)3, top_faces);
	tri_quad left(point3(555, 0, 0), vec3(0, 0, 555), vec3(0, 555, 0), (const uint)1, left_faces);
	tri_quad right(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), (const uint)6, right_faces);

	//light
	point3 center = vec3(555.f / 2.f, 554.f, 555.f / 2.f);
	float width = 100.f;
	float depth = 100.f;
	float height = 150.f;
	float margin = 1.5f;
	point3 Q = point3((center.x() + width / 2.f), center.y(), (center.z() + depth / 2.f));
	tri_quad light(Q, vec3(-width, 0, 0), vec3(0, 0, -depth), (const uint)4, light_faces);
	//tri_quad light = tri_quad(point3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), d_mat_list[4], light_faces);

	/*
	hittable** box_1_tris = &d_list[12];
	hittable** box_2_tris = &d_list[24];
	hittable** air_tris = &d_list[36];
	hittable** box_3_tris = &d_list[48];
	*/

	//others
	tri_box box1(point3(0.f, 0.f, 0.f), point3(165.f, 330.f, 165.f), (const uint)5, &d_list[12]);
	box1.rotate(degrees_to_radians(25.f), transform::AXIS::Y, false);
	box1.translate(vec3(265.f, 0.f, 295.f));


	tri_box box2(point3(0.f, 0.f, 0.f), point3(165.f, 165.f, 165.f), (const uint)0, &d_list[24]);
	box2.rotate(degrees_to_radians(-18.f), transform::AXIS::Y, false);
	box2.translate(vec3(130.f, 0.f, 65.f));



	pyramid pyr(point3(165.f, 166.f, 0.f), vec3(-165.f, 0.f, 0.f), vec3(0.f, 0.f, 165.f), vec3(0.f, 165.f, 0.f), (const uint)2, &d_list[36]);
	pyr.rotate(degrees_to_radians(-18.f), transform::AXIS::Y, false);
	pyr.translate(vec3(130.f, 0.f, 65.f));

}

__device__
void scene::device_prism_test(tri** d_list, material* d_mat_list) {

	d_mat_list[0] = material::lambertian(color(.73, .73, .73)); //white
	d_mat_list[1] = material::emissive(color(1, 1, 1), 5); //light
	d_mat_list[2] = material::dielectric(dev_flint_glass_b, dev_flint_glass_c); //flint glass

	tri** bottom_faces = &d_list[0];
	tri** top_faces = &d_list[2];
	tri** back_faces = &d_list[4];
	tri** left_faces = &d_list[6];
	tri** right_faces = &d_list[8];
	tri** light_faces = &d_list[10];

	tri_quad bottom(point3(0, 0, 0), vec3(0, 0, 555), vec3(555, 0, 0), (const uint)0, bottom_faces);
	tri_quad back(point3(0, 0, 555.f), vec3(0, 555, 0), vec3(555, 0, 0), (const uint)0, back_faces);
	tri_quad top(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), (const uint)0, top_faces);
	tri_quad left(point3(555, 0, 0), vec3(0, 0, 555), vec3(0, 555, 0), (const uint)0, left_faces);
	tri_quad right(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), (const uint)0, right_faces);

	//light
	point3 center = vec3(555.f / 2.f, 554.f, 555.f / 2.f);
	//float width = 130.f;
	float width = 100.f;
	//float depth = 105.f;
	float depth = 100.f;
	float height = 150.f;
	float margin = 1.5f;

	point3 Q = point3((center.x() + width / 2.f), center.y(), (center.z() + depth / 2.f));
	tri_quad light(Q, vec3(-width, 0, 0), vec3(0, 0, -depth), 1, light_faces);



	tri** prism_sides = &d_list[12];
	float prism_width = 165.f;
	float prism_height = 200.f;

	prism p(point3(center.x() - width / 2.f, center.y() - 1.f, center.z() - prism_height / 2.f), point3(0.f, -prism_width, 0.f), point3((prism_width * sqrt(3.f)) / 2.f, -prism_width / 2.f, 0.f), point3(0.f, 0.f, 200.f), (const uint)2, prism_sides);
	p.rotate(degrees_to_radians(10.f), transform::AXIS::Y, true);

}

__device__
void scene::device_different_mats_world(tri** d_list, material* d_mat_list) {
	d_mat_list[0] =  material::lambertian(color(.65f, .05f, .05f)); //red
	d_mat_list[1] = material::lambertian(color(.12f, .45f, .15f)); //green
	d_mat_list[2] =  material::dielectric(dev_flint_glass_b, dev_flint_glass_c);
	d_mat_list[3] =  material::lambertian(color(.73f, .73f, .73f)); //white
	d_mat_list[4] =  material::emissive(color(1.f, 1.f, 1.f), 5.f); //light
	d_mat_list[5] =  material::metallic(color(.5f, .5f, .5f), 0.3f); //metal
	d_mat_list[6] = material::lambertian(color(.12f, .15f, .45f)); //blue
	d_mat_list[7] =  material::dielectric(dev_BK7_b, dev_BK7_c);
	d_mat_list[8] = material::metallic(color(.7f, .7f, .7f), 0.8f); //metal

	tri** bottom_faces = &d_list[0];
	tri** top_faces = &d_list[2];
	tri** back_faces = &d_list[4];
	tri** left_faces = &d_list[6];
	tri** right_faces = &d_list[8];
	tri** light_faces = &d_list[10];

	//walls
	tri_quad bottom(point3(0, 0, 0), vec3(0, 0, 555), vec3(555, 0, 0), (const uint)6, bottom_faces);
	tri_quad back(point3(0, 0, 555.f), vec3(0, 555, 0), vec3(555, 0, 0), (const uint)1, back_faces);
	tri_quad top(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), (const uint)2, top_faces);
	tri_quad left(point3(555, 0, 0), vec3(0, 0, 555), vec3(0, 555, 0), (const uint)8, left_faces);
	tri_quad right(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), (const uint)5, right_faces);

	//light
	point3 center = vec3(555.f / 2.f, 554.f, 555.f / 2.f);
	float width = 100.f;
	float depth = 100.f;
	float height = 150.f;
	float margin = 1.5f;
	point3 Q = point3((center.x() + width / 2.f), center.y(), (center.z() + depth / 2.f));
	tri_quad light(Q, vec3(-width, 0, 0), vec3(0, 0, -depth), (const uint)4, light_faces);

	//others
	const uint mat_arr_1[6] = { 3, 8, 0, 1, 2, 3};
	tri_box box1(point3(0.f, 0.f, 0.f), point3(165.f, 330.f, 165.f), mat_arr_1, &d_list[12]);
	box1.rotate(degrees_to_radians(25.f), transform::AXIS::Y, false);
	box1.translate(vec3(265.f, 0.f, 295.f));

	const uint mat_arr_2[6] = {7, 6, 8, 7, 1, 2};
	tri_box box2(point3(0.f, 0.f, 0.f), point3(165.f, 165.f, 165.f), mat_arr_2, &d_list[24]);
	box2.rotate(degrees_to_radians(-18.f), transform::AXIS::Y, false);
	box2.translate(vec3(130.f, 0.f, 65.f));



	pyramid pyr(point3(165.f, 166.f, 0.f), vec3(-165.f, 0.f, 0.f), vec3(0.f, 0.f, 165.f), vec3(0.f, 165.f, 0.f), (const uint)2, &d_list[36]);
	pyr.rotate(degrees_to_radians(-18.f), transform::AXIS::Y, false);
	pyr.translate(vec3(130.f, 0.f, 65.f));
}

__host__
void scene::init_world_parameters(uint world_selector, int* world_size_ptr, int* n_materials_ptr) {

	/*
	 * select correct parameters (hard-coded) based on world selector
	 */

	switch (world_selector) {

	case PRISM:
		//prism test
		*world_size_ptr = 10 + 2 + 8; //walls + light + prism
		*n_materials_ptr = 3; //white + emission + dielectric
		break;

	case TRIS:
		//tris
		*world_size_ptr = 10 + 2 + 12 + 12 + 6; //walls + light + box1 + box2 + pyramid + prism
		*n_materials_ptr = 9;
		break;

	case CORNELL:
	default:

		//cornell
		*world_size_ptr = 10 + 2 + 12 + 12 + 6; //walls + light + box1 + box2 + pyramid + prism
		*n_materials_ptr = 7;
		break;
	}
}

__host__
camera_builder scene::cornell_box_camera_builder() {
	float vfov = 40.0f;
	point3 lookfrom = point3(278, 278, -800);
	point3 lookat = point3(278, 278, 0);
	vec3 vup = vec3(0, 1, 0);
	float defocus_angle = 0.0f;
	float focus_dist = 10.0f;
	color background = color(0.0f, 0.0f, 0.0f);
	//color background = color(0.70, 0.80, 1.00);

	return camera_builder().
		setVfov(vfov).
		setLookfrom(lookfrom).
		setVup(vup).
		setLookat(lookat).
		setDefocusAngle(defocus_angle).
		setFocusDist(focus_dist).
		setBackground(background);
}

__host__
camera_builder scene::different_mats_camera_builder() {
	float vfov = 40.0f;
	point3 lookfrom = point3(278, 278, -800);
	point3 lookat = point3(278, 278, 0);
	vec3 vup = vec3(0, 1, 0);
	float defocus_angle = 0.0f;
	float focus_dist = 10.0f;
	color background = color(0.0f, .0f, 0.f);
	//color background = color(0.7f, .7f, 1.f);

	return camera_builder().
		setVfov(vfov).
		setLookfrom(lookfrom).
		setVup(vup).
		setLookat(lookat).
		setDefocusAngle(defocus_angle).
		setFocusDist(focus_dist).
		setBackground(background);
}

__host__
camera_builder scene::prism_test_camera_builder() {
	float vfov = 40.0f;
	point3 lookfrom = point3(278, 278, -800);
	point3 lookat = point3(278, 278, 0);
	vec3 vup = vec3(0, 1, 0);
	float defocus_angle = 0.0f;
	float focus_dist = 10.0f;
	color background = color(0.0f, 0.0f, 0.0f);
	//color background = color(0.70, 0.80, 1.00);

	return camera_builder().
		setVfov(vfov).
		setLookfrom(lookfrom).
		setVup(vup).
		setLookat(lookat).
		setDefocusAngle(defocus_angle).
		setFocusDist(focus_dist).
		setBackground(background);
}

__host__
bool scene::create_bvh(tri** d_world, size_t world_size, bvh** d_bvh) {
	bool h_success;
	bool* d_success;

	checkCudaErrors(cudaMalloc((void**)&d_success, sizeof(bool)));
	create_bvh_kernel << <1, 1 >> > (d_world, world_size, d_bvh, d_success);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(&h_success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_success));
	checkCudaErrors(cudaDeviceSynchronize());

	return h_success;
}

__host__
void scene::create_world(uint world_selector, tri** d_list, material* d_mat_list, int* world_size, int* n_materials, float* dev_sRGBToSpectrum_Data) {
	create_world_kernel << <1, 1 >> > (world_selector, d_list, d_mat_list, world_size, n_materials, dev_sRGBToSpectrum_Data);
}

__host__
void scene::free_world(tri** d_list, bvh** dev_bvh, material* d_mat_list, int world_size,
	int n_materials) {
	free_world_kernel << <1, 1 >> > (d_list, world_size, n_materials, dev_bvh);

}

const result scene_manager::init_world() {
	/*
 * Allocate memory on GPU for hittables and materials, initialize their contents based on a world selector
 * then build a BVH
 */

	int* dev_n_materials_ptr = nullptr;
	int* dev_world_size_ptr = nullptr;

	//select the correct values for size and #materials
	init_world_parameters(selected_world, h_world_size_ptr, h_n_materials_ptr);

	//copy parameters to device memory so they can be modified from device code
	checkCudaErrors(cudaMalloc((void**)&dev_n_materials_ptr, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&dev_world_size_ptr, sizeof(int)));
	checkCudaErrors(cudaMemcpy(dev_n_materials_ptr, h_n_materials_ptr, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_world_size_ptr, h_world_size_ptr, sizeof(int), cudaMemcpyHostToDevice));

	//allocate space for world
	checkCudaErrors(cudaMalloc((void**)&dev_world, *(h_world_size_ptr) * sizeof(tri*)));
	if (*(h_world_size_ptr) > 0 && dev_world == nullptr) {
		return { false, "Not enough memory on device for dev_list\n" };
	}

	//allocate space for materials
	checkCudaErrors(cudaMalloc((void**)&dev_mat_list, (*h_n_materials_ptr) * sizeof(material)));
	if ((*h_n_materials_ptr) > 0 && dev_mat_list == nullptr) {
		return { false, "Not enough memory on device for dev_mat_list\n" };
	}

	//allocate space for bvh
	checkCudaErrors(cudaMalloc((void**)&dev_bvh, sizeof(bvh*)));
	if (dev_bvh == nullptr) {
		return { false, "Not enough memory on device for BVH\n" };
	}

	//copy constant to device global memory (cannot use constant memory since table is too big)
	float* dev_ColorToSpectrum_Data;
	checkCudaErrors(cudaMalloc((void**)&dev_ColorToSpectrum_Data, 3 * 64 * 64 * 64 * 3 * sizeof(float)));
	if (dev_ColorToSpectrum_Data == nullptr) {
		return { false, "Not enough memory on device for dev_ColorToSpectrum_Data\n" };
	}
	checkCudaErrors(
		cudaMemcpy(dev_ColorToSpectrum_Data, sRGBToSpectrumTable_Data, 3 * 64 * 64 * 64 * 3 * sizeof(float),
			cudaMemcpyHostToDevice));



	//build hittables and materials
	create_world(selected_world, dev_world, dev_mat_list, dev_world_size_ptr, dev_n_materials_ptr, dev_ColorToSpectrum_Data);
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
		return { false, "Error building BVH\n" };
	}

	auto lc = log_context::getInstance();
	lc->add_entry("scene type", sceneIdToStr[selected_world]);
	lc->add_entry("# primitives", *h_world_size_ptr);
	lc->add_entry("# materials", *h_n_materials_ptr);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	world_inited = true;
	return { true, "World created" };
}

__host__
void scene_manager::init_camera() {
	camera_builder cam_builder;
	switch (selected_world) {

	case PRISM:
		cam_builder = prism_test_camera_builder();
		break;

	case TRIS:
		cam_builder = different_mats_camera_builder();
		break;

	case CORNELL:
	default:
		cam_builder = cornell_box_camera_builder();
		break;
	}
	auto pm = param_manager::getInstance();
	cam = cam_builder.getCamera();

	auto lc = log_context::getInstance();
	lc->add_entry("image width", cam.getImageWidth());
	lc->add_entry("image height", cam.getImageHeight());

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
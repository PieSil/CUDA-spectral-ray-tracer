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
void create_world_kernel(uint world_selector, tri** d_list, material** d_mat_list, int* world_size, int* n_materials, float* dev_sRGBToSpectrum_Data) {

	/*
	 * initialize hittables and materials based on a world selector
	 */

	if (threadIdx.x == 0 && blockIdx.x == 0) {
		switch (world_selector) {
			/*
		case RANDOM:
			device_random_world(d_list, d_mat_list, world_size, n_materials);
			break;
		case QUADS:
			device_quad_world(d_list, d_mat_list);
			break;

		case SIMPLE_LIGHT:
			device_simple_light(d_list, d_mat_list);
			break;
			 */

		case PRISM:
			device_prism_test(d_list, d_mat_list);
			break;

			/*
		case SPHERES:
			device_3_spheres(d_list, d_mat_list);
			break;
			 */

		case TRIS:
			device_tri_world(d_list, d_mat_list);
			break;

		case CORNELL:
		default:
			device_cornell_box(d_list, d_mat_list);
			device_cornell_box(d_list, d_mat_list);
			/*
			device_random_world(d_list, d_mat_list, world_size, n_materials);
			 */
		}

		/*
		 * precompute reflectance and emittance spectrum
		 */



		for (int i = 0; i < *n_materials; i++) {
			d_mat_list[i]->compute_spectral_distr(dev_sRGBToSpectrum_Data);
		}

	}
}

__global__ void
free_world_kernel(tri** d_list, material** d_mat_list, int world_size, int n_materials, bvh** dev_bvh) {

	if (d_list != nullptr && threadIdx.x == 0 && blockIdx.x == 0) {

		for (int i = 0; i < n_materials; i++) {
			delete* (d_mat_list + i);
		}

		for (int i = 0; i < world_size; i++) {
			delete* (d_list + i);
		}

		delete* dev_bvh;
	}
}


//__device__
//void scene::device_random_world(hittable** d_list, material** d_mat_list, int* world_size, int* n_materials) {
//	curandState _rand_state = curandState();
//	curandState* rand_state = &_rand_state;
//	curand_init(1999, 0, 0, rand_state);
//	int void_positions = 0;
//
//	for (int a = -HALF_N_RANDOM_I; a < HALF_N_RANDOM_I; a++) {
//		for (int b = -HALF_N_RANDOM_J; b < HALF_N_RANDOM_J; b++) {
//			auto choose_mat = cuda_random_float(rand_state);
//			point3 center(float(a) + 0.9f * cuda_random_float(rand_state), 0.2f, float(b) + 0.9f * cuda_random_float(rand_state));
//			int list_idx = (a + HALF_N_RANDOM_I) * 2 * HALF_N_RANDOM_J + (b + HALF_N_RANDOM_J) - void_positions;
//
//			if ((center - point3(4, 0.2, 0)).length() > 0.9) {
//
//				if (choose_mat < 0.8f) {
//					// diffuse
//					auto albedo = color::random(rand_state) * color::random(rand_state);
//					*(d_mat_list + list_idx) = new material();
//					**(d_mat_list + list_idx) = material::lambertian(albedo);
//					*(d_list + list_idx) = new sphere(center, 0.2f, *(d_mat_list + list_idx));
//				}
//				else if (choose_mat < 0.95f) {
//					// metal
//					//auto albedo = color::random(rand_state);
//					auto albedo = color::random(0.5f, 1.0f, rand_state);
//					auto fuzz = cuda_random_float(0.0f, 0.5f, rand_state);
//					*(d_mat_list + list_idx) = new material();
//					**(d_mat_list + list_idx) = material::metallic(albedo, fuzz);
//					*(d_list + list_idx) = new sphere(center, 0.2f, *(d_mat_list + list_idx));
//				}
//				else {
//					// glass
//					*(d_mat_list + list_idx) = new material();
//					**(d_mat_list + list_idx) = material::dielectric_const(1.5f);
//					*(d_list + list_idx) = new sphere(center, 0.2f, *(d_mat_list + list_idx));
//				}
//
//				/*if (*(d_list+list_idx) == nullptr)
//					printf("list[%d] is nullptr\n");
//				if (*(d_mat_list+list_idx) == nullptr)
//					printf("material[%d] is nullptr\n");*/
//
//			}
//			else {
//				void_positions++;
//			}
//		}
//	}
//
//	*(d_mat_list + N_RANDOM_MATERIALS - void_positions) = new material();
//	**(d_mat_list + N_RANDOM_MATERIALS - void_positions) = material::lambertian(color(0.5, 0.5, 0.5));
//	*(d_list + N_RANDOM_SPHERES - void_positions) = new sphere(point3(0, -1000, 0), 1000, *(d_mat_list + N_RANDOM_MATERIALS - void_positions));
//
//	*(d_mat_list + N_RANDOM_MATERIALS - void_positions + 1) = new material();
//	**(d_mat_list + N_RANDOM_MATERIALS - void_positions + 1) = material::dielectric_const(1.5f);
//	*(d_list + N_RANDOM_SPHERES - void_positions + 1) = new sphere(point3(0, 1, 0), 1.0, *(d_mat_list + N_RANDOM_MATERIALS - void_positions + 1));
//
//	*(d_mat_list + N_RANDOM_MATERIALS - void_positions + 2) = new material();
//	**(d_mat_list + N_RANDOM_MATERIALS - void_positions + 2) = material::lambertian(color(0.4, 0.2, 0.1));
//	*(d_list + N_RANDOM_SPHERES - void_positions + 2) = new sphere(point3(-4, 1, 0), 1.0, *(d_mat_list + N_RANDOM_MATERIALS - void_positions + 2));
//
//	*(d_mat_list + N_RANDOM_MATERIALS - void_positions + 3) = new material();
//	**(d_mat_list + N_RANDOM_MATERIALS - void_positions + 3) = material::metallic(color(0.7, 0.6, 0.5), 0.0);
//	*(d_list + N_RANDOM_SPHERES - void_positions + 3) = new sphere(point3(4, 1, 0), 1.0, *(d_mat_list + N_RANDOM_MATERIALS - void_positions + 3));
//
//	if (world_size != nullptr)
//		*world_size = *world_size - void_positions;
//
//	if (n_materials != nullptr)
//		*n_materials = *n_materials - void_positions;
//
//	delete rand_state;
//}

//__device__
//void scene::device_quad_world(hittable** d_list, material** d_mat_list) {
//	d_mat_list[0] = new material();
//	*d_mat_list[0] = material::lambertian(color(1.0, 0.2, 0.2));
//	d_mat_list[1] = new material();
//	*d_mat_list[1] = material::lambertian(color(0.2, 1.0, 0.2));
//	d_mat_list[2] = new material();
//	*d_mat_list[2] = material::lambertian(color(0.2, 0.2, 1.0));
//	d_mat_list[3] = new material();
//	*d_mat_list[3] = material::lambertian(color(1.0, 0.5, 0.0));
//	d_mat_list[4] = new material();
//	*d_mat_list[4] = material::lambertian(color(0.2, 0.8, 0.8));
//
//	d_list[0] = new quad(point3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0), d_mat_list[0]);
//	d_list[1] = new quad(point3(-2, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0), d_mat_list[1]);
//	d_list[2] = new quad(point3(3, -2, 1), vec3(0, 0, 4), vec3(0, 4, 0), d_mat_list[2]);
//	d_list[3] = new quad(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), d_mat_list[3]);
//	d_list[4] = new quad(point3(-2, -3, 5), vec3(4, 0, 0), vec3(0, 0, -4), d_mat_list[4]);
//}

//__device__
//void scene::device_simple_light(hittable** d_list, material** d_mat_list) {
//
//	d_mat_list[0] = new material();
//	*(d_mat_list[0]) = material::lambertian(color(.4f, 1.0f, .2f));
//
//	d_mat_list[1] = new material();
//	//*(d_mat_list[1]) = material::metallic(color(.5f, .5f, .5f), .5f);
//	//*(d_mat_list[1]) = material::lambertian(color(.1f, .5f, .7f));
//	*(d_mat_list[1]) = material::dielectric_const(1.5f);
//
//	d_mat_list[3] = new material();
//	*(d_mat_list[3]) = material::dielectric_const(1.0f / 1.5f);
//	//*(d_mat_list[3]) = material::lambertian(color(1.0, 0.0f, 1.0f));
//
//	d_mat_list[2] = new material();
//	*(d_mat_list[2]) = material::emissive(color(1.0f, 1.0f, 1.0f), 5.0f);
//
//	d_list[0] = new sphere(point3(0, -1000, 0), 1000, d_mat_list[0]); //ground
//	d_list[1] = new sphere(point3(0, 2, 0), 2, d_mat_list[1]); //sphere
//	d_list[3] = new sphere(point3(0, 2, 0), 1.9, d_mat_list[3]); //air bubble
//	d_list[2] = new quad(point3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), d_mat_list[2]); //light
//}

__device__
void scene::device_cornell_box(tri** d_list, material** d_mat_list) {
	d_mat_list[0] = new material();
	*d_mat_list[0] = material::lambertian(color(.65f, .05f, .05f)); //red
	d_mat_list[1] = new material();
	*d_mat_list[1] = material::lambertian(color(.12f, .45f, .15f)); //green
	d_mat_list[2] = new material();
	*d_mat_list[2] = material::dielectric(dev_flint_glass_b, dev_flint_glass_c);
	d_mat_list[3] = new material();
	*d_mat_list[3] = material::lambertian(color(.73f, .73f, .73f)); //white
	d_mat_list[4] = new material();
	*d_mat_list[4] = material::emissive(color(1.f, 1.f, 1.f), 5.f); //light
	d_mat_list[5] = new material();
	*d_mat_list[5] = material::metallic(color(.5f, .5f, .5f), 0.3f); //metal
	d_mat_list[6] = new material();
	*d_mat_list[6] = material::lambertian(color(.12f, .15f, .45f)); //blue

	tri** bottom_faces = &d_list[0];
	tri** top_faces = &d_list[2];
	tri** back_faces = &d_list[4];
	tri** left_faces = &d_list[6];
	tri** right_faces = &d_list[8];
	tri** light_faces = &d_list[10];

	//walls
	tri_quad bottom = tri_quad(point3(0, 0, 0), vec3(0, 0, 555), vec3(555, 0, 0), d_mat_list[3], bottom_faces);
	tri_quad back = tri_quad(point3(0, 0, 555.f), vec3(0, 555, 0), vec3(555, 0, 0), d_mat_list[3], back_faces);
	tri_quad top = tri_quad(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), d_mat_list[3], top_faces);
	tri_quad left = tri_quad(point3(555, 0, 0), vec3(0, 0, 555), vec3(0, 555, 0), d_mat_list[1], left_faces);
	tri_quad right = tri_quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), d_mat_list[6], right_faces);

	//light
	point3 center = vec3(555.f / 2.f, 554.f, 555.f / 2.f);
	float width = 100.f;
	float depth = 100.f;
	float height = 150.f;
	float margin = 1.5f;
	point3 Q = point3((center.x() + width / 2.f), center.y(), (center.z() + depth / 2.f));
	tri_quad light = tri_quad(Q, vec3(-width, 0, 0), vec3(0, 0, -depth), d_mat_list[4], light_faces);
	//tri_quad light = tri_quad(point3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), d_mat_list[4], light_faces);

	/*
	hittable** box_1_tris = &d_list[12];
	hittable** box_2_tris = &d_list[24];
	hittable** air_tris = &d_list[36];
	hittable** box_3_tris = &d_list[48];
	*/

	//others
	tri_box box1 = tri_box(point3(0.f, 0.f, 0.f), point3(165.f, 330.f, 165.f), d_mat_list[5], &d_list[12]);
	box1.rotate(degrees_to_radians(25.f), transform::AXIS::Y, false);
	box1.translate(vec3(265.f, 0.f, 295.f));


	tri_box box2 = tri_box(point3(0.f, 0.f, 0.f), point3(165.f, 165.f, 165.f), d_mat_list[0], &d_list[24]);
	box2.rotate(degrees_to_radians(-18.f), transform::AXIS::Y, false);
	box2.translate(vec3(130.f, 0.f, 65.f));



	pyramid pyr(point3(165.f, 166.f, 0.f), vec3(-165.f, 0.f, 0.f), vec3(0.f, 0.f, 165.f), vec3(0.f, 165.f, 0.f), d_mat_list[2], &d_list[36]);
	pyr.rotate(degrees_to_radians(-18.f), transform::AXIS::Y, false);
	pyr.translate(vec3(130.f, 0.f, 65.f));

}

__device__
void scene::device_prism_test(tri** d_list, material** d_mat_list) {

	d_mat_list[0] = new material();
	*d_mat_list[0] = material::lambertian(color(.73, .73, .73)); //white
	d_mat_list[1] = new material();
	*d_mat_list[1] = material::emissive(color(1, 1, 1), 5); //light
	d_mat_list[2] = new material();
	//*d_mat_list[2] = material::normal_test(color(0.1f, 0.1f, 0.7f));
	*d_mat_list[2] = material::dielectric(dev_flint_glass_b, dev_flint_glass_c);
	//*d_mat_list[2] = material::dielectric(dev_flint_glass_b, dev_flint_glass_c); //glass
	//*d_mat_list[2] = material::dielectric(dev_BK7_b, dev_BK7_c); //glass

	tri** bottom_faces = &d_list[0];
	tri** top_faces = &d_list[2];
	tri** back_faces = &d_list[4];
	tri** left_faces = &d_list[6];
	tri** right_faces = &d_list[8];
	tri** light_faces = &d_list[10];

	tri_quad bottom = tri_quad(point3(0, 0, 0), vec3(0, 0, 555), vec3(555, 0, 0), d_mat_list[0], bottom_faces);
	tri_quad back = tri_quad(point3(0, 0, 555.f), vec3(0, 555, 0), vec3(555, 0, 0), d_mat_list[0], back_faces);
	tri_quad top = tri_quad(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), d_mat_list[0], top_faces);
	tri_quad left = tri_quad(point3(555, 0, 0), vec3(0, 0, 555), vec3(0, 555, 0), d_mat_list[0], left_faces);
	tri_quad right = tri_quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), d_mat_list[0], right_faces);

	//walls
	/*
	d_list[0] = new quad(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), d_mat_list[0]);
	d_list[1] = new quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), d_mat_list[0]);
	d_list[2] = new quad(point3(0, 0, 0), vec3(0, 0, 555), vec3(555, 0, 0), d_mat_list[0]);
	d_list[3] = new quad(point3(555, 555, 555), vec3(0, 0, -555), vec3(-555, 0, 0), d_mat_list[0]);
	d_list[4] = new quad(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), d_mat_list[0]);
	*/

	//light
	point3 center = vec3(555.f / 2.f, 554.f, 555.f / 2.f);
	//float width = 130.f;
	float width = 100.f;
	//float depth = 105.f;
	float depth = 100.f;
	float height = 150.f;
	float margin = 1.5f;

	point3 Q = point3((center.x() + width / 2.f), center.y(), (center.z() + depth / 2.f));
	//d_list[5] = new quad(Q, vec3(-width, 0, 0), vec3(0, 0, -depth), d_mat_list[1]); //light
	tri_quad light = tri_quad(Q, vec3(-width, 0, 0), vec3(0, 0, -depth), d_mat_list[1], light_faces);
	/*
	hittable** light_back_sides = &d_list[12];
	hittable** light_front_sides = &d_list[14];
	hittable** light_right_sides = &d_list[16];
	hittable** light_left_sides = &d_list[18];

	tri_quad light_back(point3(Q.x() + margin, Q.y() + 2.f, (Q.z() - margin) - (depth + 2.f * margin)), vec3(0, -height, 0), vec3(-(width + 2.f * margin), 0, 0), d_mat_list[0], light_back_sides);
	tri_quad light_front(point3(Q.x() + margin, Q.y() + 2.f, Q.z() - margin), vec3(-(width + 2.f * margin), 0, 0), vec3(0, -height, 0), d_mat_list[0], light_front_sides);
	tri_quad light_right(point3((Q.x() + margin), Q.y(), Q.z() - margin), vec3(0.f, -height, 0), vec3(0.f, 0.f, -(depth + 2.f * margin)), d_mat_list[0], light_right_sides);
	tri_quad light_left(point3(Q.x() + margin - (width + 2.f * margin), Q.y(), Q.z() - margin), vec3(0.f, 0.f, -(depth + 2.f * margin)), vec3(0.f, -100.f, 0), d_mat_list[0], light_left_sides);
	*/

	/*
	d_list[6] = new quad(point3(Q.x()+margin, Q.y()+2.f, (Q.z()-margin) - (depth+2.f*margin)), vec3(0, -height, 0), vec3(-(width+2.f*margin), 0, 0), d_mat_list[0]); //back wall
	d_list[7] = new quad(point3(Q.x()+margin, Q.y()+2.f, Q.z()-margin), vec3(-(width+2.f*margin), 0, 0), vec3(0, -height, 0), d_mat_list[0]); //front wall
	d_list[8] = new quad(point3((Q.x()+margin) , Q.y(), Q.z()-margin), vec3(0.f, -height, 0), vec3(0.f, 0.f, -(depth + 2.f*margin)), d_mat_list[0]); //right wall
	d_list[9] = new quad(point3(Q.x()+margin - (width + 2.f*margin), Q.y(), Q.z() - margin), vec3(0.f, 0.f, -(depth + 2.f * margin)), vec3(0.f, -100.f, 0), d_mat_list[0]); //left wall
	*/

	tri** prism_sides = &d_list[12];
	float prism_width = 165.f;
	float prism_height = 200.f;
	//tri_box prism = tri_box(point3(0.f, 0.f, 0.f), point3(165.f, 165.f, 165.f), d_mat_list[2], prism_sides);
	// Q = (center - width/2.f, center.y - 1.f, center.z - prism_height);
	// u = (0.f, -prims_width, 0.f)
	// v = ((prism_width * sqrt(3.f)) / 2.f, prims_width/2.f, 0.f)
	// w = (0.f, 0.f, prism_height);
	// 
	//prism p = prism(point3(0.f, 0.f, 0.f), point3(prism_width, 0.f, 0.f), point3(prism_width / 2.f, 0.f, (prism_width * sqrt(3.f)) / 2.f), point3(0.f, -200.f, 0.f), d_mat_list[2], prism_sides);
	/*NICE*/ prism p = prism(point3(center.x() - width / 2.f, center.y() - 1.f, center.z() - prism_height / 2.f), point3(0.f, -prism_width, 0.f), point3((prism_width * sqrt(3.f)) / 2.f, -prism_width / 2.f, 0.f), point3(0.f, 0.f, 200.f), d_mat_list[2], prism_sides);
	//prism p = prism(point3(center.x() - width / 2.f, center.y() - 1.f, center.z() - prism_height / 2.f), point3(0.f, -prism_width, 0.f), point3((prism_width * sqrt(3.f)) / 2.f, -prism_width / 2.f, 0.f), point3(0.f, 0.f, 200.f), d_mat_list[2], prism_sides);
	p.rotate(degrees_to_radians(10.f), transform::AXIS::Y, true);

	/*
	p.rotate(degrees_to_radians(90.f), transform::AXIS::X, false);
	p.rotate(degrees_to_radians(-30.f), transform::AXIS::Z, false);
	p.rotate(degrees_to_radians(10.f), transform::AXIS::Y, false);
	p.translate(vec3(277.5f - (prism_width/2.f), 0.f , 0.f), false);
	//p.rotate(degrees_to_radians(-45.f), transform::AXIS::Y, false);

	p.translate(vec3(0.f, 450.f, 277.5f), true);
	*/



	/*
	transform::translate_box(prism, vec3(343.f - width - (165.f - width) / 2.f, 553.f - (sqrt(2.0f) * 165.f), 332.f - depth - (165.f - depth) / 2.f), false);
	transform::rotate_box(prism, degrees_to_radians(-45.f), transform::AXIS::Y, false);
	transform::rotate_box(prism, degrees_to_radians(-90.f), transform::AXIS::X, false);
	transform::translate_box(prism, vec3(-50.f, 0.f, 0.f), true);
	*/

}

__device__
void scene::device_tri_world(tri** d_list, material** d_mat_list) {
	d_mat_list[0] = new material();
	*d_mat_list[0] = material::lambertian(color(.65f, .05f, .05f)); //red
	//*d_mat_list[0] = material::normal_test(color(.65f, .05f, .05f)); //red
	d_mat_list[1] = new material();
	//*d_mat_list[1] = material::lambertian(color(.05f, .65f, .05f)); //green
	*d_mat_list[1] = material::lambertian(color(.05f, .65f, .05f)); //green
	d_mat_list[2] = new material();
	//*d_mat_list[2] = material::lambertian(color(.05f, .05f, .65f)); //blue
	//*d_mat_list[2] = material::dielectric_const(1.5f); //glass
	*d_mat_list[2] = material::dielectric(dev_flint_glass_b, dev_flint_glass_c);
	d_mat_list[3] = new material();
	//*d_mat_list[3] = material::lambertian(color(.75f, .75f, .75f)); //white
	*d_mat_list[3] = material::lambertian(color(.75f, .75f, .75f)); //white
	d_mat_list[4] = new material();
	//*d_mat_list[3] = material::lambertian(color(.75f, .75f, .75f)); //white
	*d_mat_list[4] = material::emissive(color(1.f, 1.f, 1.f), 8.f); //light

	tri** bottom_faces = &d_list[0];
	tri** top_faces = &d_list[2];
	tri** back_faces = &d_list[4];
	tri** left_faces = &d_list[6];
	tri** right_faces = &d_list[8];
	tri** light_faces = &d_list[10];

	tri_quad bottom = tri_quad(point3(0, 0, 0), vec3(0, 0, 555), vec3(555, 0, 0), d_mat_list[3], bottom_faces);
	tri_quad back = tri_quad(point3(0, 0, 555.f), vec3(0, 555, 0), vec3(555, 0, 0), d_mat_list[3], back_faces);
	tri_quad top = tri_quad(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), d_mat_list[3], top_faces);
	tri_quad left = tri_quad(point3(555, 0, 0), vec3(0, 0, 555), vec3(0, 555, 0), d_mat_list[1], left_faces);
	tri_quad right = tri_quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), d_mat_list[0], right_faces);
	tri_quad light = tri_quad(point3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), d_mat_list[4], light_faces);

	/*
	hittable** box_1_tris = &d_list[12];
	hittable** box_2_tris = &d_list[24];
	hittable** air_tris = &d_list[36];
	hittable** box_3_tris = &d_list[48];
	*/

	tri_box box1 = tri_box(point3(0.f, 0.f, 0.f), point3(165.f, 330.f, 165.f), d_mat_list[3], &d_list[12]);
	box1.rotate(degrees_to_radians(15.f), transform::AXIS::Y, false);
	box1.translate(vec3(265.f, 0.f, 295.f));

	tri_box box2 = tri_box(point3(0.f, 0.f, 0.f), point3(165.f, 165.f, 165.f), d_mat_list[2], &d_list[24]);
	box2.rotate(degrees_to_radians(-18.f), transform::AXIS::Y, false);
	box2.translate(vec3(130.f, 0.f, 65.f));

	tri_box air = tri_box(box2, d_mat_list[2], &d_list[36], 1.f);
	air.flip_normals();
}

/*
__device__
void scene::device_3_spheres(tri **d_list, material** d_mat_list) {
	d_mat_list[0] = new material();
	*d_mat_list[0] = material::lambertian(color(.8f, .7f, .0f)); //ground
	d_mat_list[1] = new material();
	*d_mat_list[1] = material::lambertian(color(.1f, .2f, .5f)); //center
	d_mat_list[2] = new material();
	//*d_mat_list[2] = material::lambertian(color(.8f, .8f, .8f)); //left
	*d_mat_list[2] = material::dielectric_const(1.5f);
	d_mat_list[3] = new material();
	*d_mat_list[3] = material::lambertian(color(.8f, .6f, .2f)); //right

	d_mat_list[4] = new material();
	*d_mat_list[4] = material::dielectric_const(1.0f / 1.5f); //air

	d_list[0] = new sphere(point3(0.0, -100.5, -1.0), 100.0, d_mat_list[0]); //ground
	d_list[1] = new sphere(point3(0.0, 0.0, -1.2), 0.5, d_mat_list[1]); //center
	d_list[2] = new sphere(point3(-1.0, 0.0, -1.0), 0.5, d_mat_list[2]); //left
	d_list[3] = new sphere(point3(1.0, 0.0, -1.0), 0.5, d_mat_list[3]); //right
	d_list[4] = new sphere(point3(-1.0, 0.0, -1.0), 0.4, d_mat_list[4]); //air
}
 */

__host__
void scene::init_world_parameters(uint world_selector, int* world_size_ptr, int* n_materials_ptr) {

	/*
	 * select correct parameters (hard-coded) based on world selector
	 */

	switch (world_selector) {
		/*
	case RANDOM:
		//random world
		*world_size_ptr = RANDOM_WORLD_SIZE;
		*n_materials_ptr = RANDOM_WORLD_MATERIALS;
		break;
		 */

		 /*
	 case QUADS:
		 //quads
		 *world_size_ptr = 5;
		 *n_materials_ptr = 5;
		 break;
		  */

		  /*
	  case SIMPLE_LIGHT:
		  //light
		  *world_size_ptr = 4;
		  *n_materials_ptr = 4;
		  break;
		   */

	case PRISM:
		//prism test
		*world_size_ptr = 10 + 2 + 8; //walls + light + prism
		*n_materials_ptr = 3; //white + emission + dielectric
		break;

		/*
	case SPHERES:
		//spheres
		*world_size_ptr = 5;
		*n_materials_ptr = 5;
		break;
		 */

	case TRIS:
		//tris
		*world_size_ptr = 12 + 12 + 12 + 12;
		*n_materials_ptr = 5;
		break;

	case CORNELL:
	default:

		//cornell
		*world_size_ptr = 10 + 2 + 12 + 12 + 6; //walls + light + box1 + box2 + pyramid + prism
		*n_materials_ptr = 7;
		break;
	}
}

//__host__
//camera_builder scene::random_world_cam_builder() {
//	float vfov = 20.0f;
//	point3 lookfrom = point3(13, 2, 3);
//	point3 lookat = point3(0, 0, 0);
//	vec3 vup = vec3(0, 1, 0);
//	float defocus_angle = 0.6f;
//	float focus_dist = 10.0f;
//	color background = color(0.70, 0.80, 1.00);
//
//	return camera_builder().
//		setVfov(vfov).
//		setLookfrom(lookfrom).
//		setVup(vup).
//		setLookat(lookat).
//		setDefocusAngle(defocus_angle).
//		setFocusDist(focus_dist).
//		setBackground(background);
//}

//__host__
//camera_builder scene::quad_world_camera_builder() {
//	float vfov = 80.0f;
//	point3 lookfrom = point3(0, 0, 9);
//	point3 lookat = point3(0, 0, 0);
//	vec3 vup = vec3(0, 1, 0);
//	float defocus_angle = 0.0f;
//	float focus_dist = 10.0f;
//	color background = color(0.70, 0.80, 1.00);
//
//	return camera_builder().
//		setVfov(vfov).
//		setLookfrom(lookfrom).
//		setVup(vup).
//		setLookat(lookat).
//		setDefocusAngle(defocus_angle).
//		setFocusDist(focus_dist).
//		setBackground(background);
//}

//__host__
//camera_builder scene::simple_light_camera_builder() {
//	float vfov = 20.0f;
//	point3 lookfrom = point3(26, 3, 6);
//	//point3 lookfrom = point3(3,26,6);
//	point3 lookat = point3(0, 2, 0);
//	vec3 vup = vec3(0, 1, 0);
//	float defocus_angle = 0.0f;
//	float focus_dist = 10.0f;
//	//color expanded_b_color = color(10.0f, 180.0f, 186.0f);
//	//color expanded_b_color = color(255.0f, 255.0f, 255.0f);
//	//color background = expanded_b_color/255;
//	//color background = color(0.5f, 0.5f, 0.5f);
//	//color background = color(0.7f, 0.7f, 1.0f);
//	color background = color(0.0f, 0.0f, 0.0f);
//	//color background = color(1.00, 1.00, 1.00);
//
//	return camera_builder().
//		setVfov(vfov).
//		setLookfrom(lookfrom).
//		setVup(vup).
//		setLookat(lookat).
//		setDefocusAngle(defocus_angle).
//		setFocusDist(focus_dist).
//		setBackground(background);
//}

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
camera_builder scene::tris_camera_builder() {
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

//__host__
//camera_builder scene::spheres_camera_builder() {
//	float vfov = 80.0f;
//	point3 lookfrom = point3(0, 0, 0);
//	point3 lookat = point3(0, 0, -1);
//	vec3 vup = vec3(0, 1, 0);
//	float defocus_angle = 0.0f;
//	float focus_dist = 10.0f;
//	color background = color(0.7, 0.8, 1.0);
//
//	return camera_builder().
//		setVfov(vfov).
//		setLookfrom(lookfrom).
//		setVup(vup).
//		setLookat(lookat).
//		setDefocusAngle(defocus_angle).
//		setFocusDist(focus_dist).
//		setBackground(background);
//}

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
void scene::create_world(uint world_selector, tri** d_list, material** d_mat_list, int* world_size, int* n_materials, float* dev_sRGBToSpectrum_Data) {
	create_world_kernel << <1, 1 >> > (world_selector, d_list, d_mat_list, world_size, n_materials, dev_sRGBToSpectrum_Data);
}

__host__
void scene::free_world(tri** d_list, bvh** dev_bvh, material** d_mat_list, int world_size,
	int n_materials) {
	free_world_kernel << <1, 1 >> > (d_list, d_mat_list, world_size, n_materials, dev_bvh);

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
	checkCudaErrors(cudaMalloc((void**)&dev_mat_list, (*h_n_materials_ptr) * sizeof(material*)));
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
		/*
	case RANDOM:
		cam_builder = random_world_cam_builder();
		break;
		 */
		 /*
	 case QUADS:
		 cam_builder = quad_world_camera_builder();
		 break;
		  */
		  /*
	  case SIMPLE_LIGHT:
		  cam_builder = simple_light_camera_builder();
		  break;
		   */

	case PRISM:
		cam_builder = prism_test_camera_builder();
		break;
		/*
	case SPHERES:
		cam_builder = spheres_camera_builder();
		break;
		 */
	case TRIS:
		cam_builder = tris_camera_builder();
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
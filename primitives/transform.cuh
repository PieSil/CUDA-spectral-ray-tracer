#ifndef SPECTRAL_RT_PROJECT_TRANSFORM_CUH
#define SPECTRAL_RT_PROJECT_TRANSFORM_CUH

#include "sphere.cuh"
#include "quad.cuh"

namespace transform {
	enum AXIS {
		NONE,
		X,
		Y,
		Z
	};

	__host__ __device__
	void translate(sphere& target, vec3 dir);

	__host__ __device__
	void translate(quad& target, vec3 dir, bool reinit=true);

	__host__ __device__
	void rotate(quad& target, float theta, transform::AXIS ax, bool reinit=true, bool local=true);

	__host__ __device__
	void translate_box(box target, vec3 dir, bool reinit=true);

	__host__ __device__
	void rotate_box(box target, float theta, transform::AXIS ax, bool reinit=true, bool local = true);

	__host__ __device__
	void assign_rot_matrix(float theta, AXIS ax, float rot_matrix[9]);
}

#endif
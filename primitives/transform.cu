#include "transform.cuh"

__host__ __device__
void transform::translate(sphere& target, vec3 dir) {
	target.center += dir;
	target.set_bbox();
}

__host__ __device__
void transform::translate(quad& target, vec3 dir, bool reinit) {
	target.Q += dir;

	if (reinit)
		target.init();
}

__host__ __device__
void transform::assign_rot_matrix(float theta, AXIS ax, float rot_matrix[9]) {

	float cos_theta = cos(theta);
	float sin_theta = sin(theta);

	switch (ax) {
	case X:
		rot_matrix[4] = cos_theta;
		rot_matrix[5] = -sin_theta;
		rot_matrix[7] = sin_theta;
		rot_matrix[8] = cos_theta;
		break;

	case Y:
		rot_matrix[0] = cos_theta;
		rot_matrix[2] = sin_theta;
		rot_matrix[6] = -sin_theta;
		rot_matrix[8] = cos_theta;
		break;

	case Z:
		rot_matrix[0] = cos_theta;
		rot_matrix[1] = -sin_theta;
		rot_matrix[3] = sin_theta;
		rot_matrix[4] = cos_theta;
		break;

	default:
		break;
	}
}

__host__ __device__
void transform::rotate(quad& target, const float theta, const transform::AXIS ax, const bool reinit, const bool local) {
	float rot_matrix[9] = {1.0f, 0.0f, 0.0f,
						   0.0f, 1.0f, 0.0f,
						   0.0f, 0.0f, 1.0f};

	float cos_theta = cos(theta);
	float sin_theta = sin(theta);

	switch (ax) {
	case X:
		rot_matrix[4] = cos_theta;
		rot_matrix[5] = -sin_theta;
		rot_matrix[7] = sin_theta;
		rot_matrix[8] = cos_theta;
		break;

	case Y:
		rot_matrix[0] = cos_theta;
		rot_matrix[2] = sin_theta;
		rot_matrix[6] = -sin_theta;
		rot_matrix[8] = cos_theta;
		break;

	case Z:
		rot_matrix[0] = cos_theta;
		rot_matrix[1] = -sin_theta;
		rot_matrix[3] = sin_theta;
		rot_matrix[4] = cos_theta;
		break;

	default:
		break;
	}

	point3 center;
	if (local) {
		center = target.center();
		transform::translate(target, -center, false);
	}
	target.Q = vec3::matrix_mul(target.Q, rot_matrix);
	target.u = vec3::matrix_mul(target.u, rot_matrix);
	target.v = vec3::matrix_mul(target.v, rot_matrix);

	if (local) {
		transform::translate(target, center, false);
	}

	if (reinit)
		target.init();

}

__host__ __device__
void transform::translate_box(box target, vec3 dir, bool reinit) {
	for (int i = 0; i < 6; i++) {
		transform::translate(*(target.sides[i]), dir, reinit);
	}
}

__host__ __device__
void transform::rotate_box(box target, float theta, transform::AXIS ax, bool reinit, bool local) {
	point3 to_origin = -target.getCenter();

	if (local)
		transform::translate_box(target, to_origin, false);

	for (int i = 0; i < 6; i++) {
		transform::rotate(*(target.sides[i]), theta, ax, false, false);
	}

	if (local)
		transform::translate_box(target, -to_origin, reinit);
}

#include "tri_quad.cuh"

__device__
void tri_quad::translate(const vec3 dir, const bool reinit) {
	halves[0]->translate(dir, reinit);
	halves[1]->translate(dir, reinit);
}

__device__
void tri_quad::rotate(const float theta, transform::AXIS ax, bool reinit, bool local) {
	point3 center_point;
	if (local) {
		center_point = center();
		translate(-center_point, false);
	}

	halves[0]->rotate(theta, ax, false, false);
	halves[1]->rotate(theta, ax, false, false);

	if (local)
		translate(center_point, false);

	if (reinit)
		init();
}
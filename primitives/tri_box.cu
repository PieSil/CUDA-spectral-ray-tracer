#include "tri_box.cuh"

__device__
void tri_box::translate(vec3 dir, bool reinit) {

	sides[0].translate(dir, reinit);
	sides[1].translate(dir, reinit);
	sides[2].translate(dir, reinit);
	sides[3].translate(dir, reinit);
	sides[4].translate(dir, reinit);
	sides[5].translate(dir, reinit);
}

__device__
void tri_box::rotate(float theta, transform::AXIS ax, bool reinit, bool local) {
	point3 center_point;

	if (local) {
		center_point = center();
		translate(-center_point, false);
	}

	sides[0].rotate(theta, ax, false, false);
	sides[1].rotate(theta, ax, false, false);
	sides[2].rotate(theta, ax, false, false);
	sides[3].rotate(theta, ax, false, false);
	sides[4].rotate(theta, ax, false, false);
	sides[5].rotate(theta, ax, false, false);

	if (local)
		translate(center_point, false);

	if (reinit)
		init();
}
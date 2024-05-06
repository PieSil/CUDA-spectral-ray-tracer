#include "prism.cuh"

__device__
prism prism::translate(const vec3 dir, const bool reinit) {
	base[0]->translate(dir, reinit);
	base[1]->translate(dir, reinit);

	sides[0].translate(dir, reinit);
	sides[1].translate(dir, reinit);
	sides[2].translate(dir, reinit);

	return *this;
}

__device__
prism prism::rotate(const float theta, const transform::AXIS ax, const bool reinit, const bool local) {
	point3 center;

	if (local) {
		center = centroid();
		translate(-center, false);
	}

	base[0]->rotate(theta, ax, false, false);
	base[1]->rotate(theta, ax, false, false);

	sides[0].rotate(theta, ax, false, false);
	sides[1].rotate(theta, ax, false, false);
	sides[2].rotate(theta, ax, false, false);

	if (local) {
		translate(center, false);
	}

	if (reinit)
		init();
}
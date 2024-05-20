#include "prism.cuh"

__device__
void prism::translate(vec3 dir, bool reinit) {
    base[0]->translate(dir, reinit);
	base[1]->translate(dir, reinit);

	sides[0].translate(dir, reinit);
	sides[1].translate(dir, reinit);
	sides[2].translate(dir, reinit);
}

__device__
void prism::rotate(float theta, transform::AXIS ax, bool reinit, bool local) {
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
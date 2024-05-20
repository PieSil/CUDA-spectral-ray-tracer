#include "pyramid.cuh"

__device__
void pyramid::translate(const vec3 dir, const bool reinit) {
	static_cast<tri*>(sides[0])->translate(dir, reinit);
	static_cast<tri*>(sides[1])->translate(dir, reinit);
	static_cast<tri*>(sides[2])->translate(dir, reinit);
	static_cast<tri*>(sides[3])->translate(dir, reinit);

	base.translate(dir, reinit);

}

__device__
void pyramid::rotate(const float theta, const transform::AXIS ax, const bool reinit, const bool local) {
	point3 center;

	if (local) {
		center = base_center();
		translate(-center, false);
	}

	static_cast<tri*>(sides[0])->rotate(theta, ax, false, false);
	static_cast<tri*>(sides[1])->rotate(theta, ax, false, false);
	static_cast<tri*>(sides[2])->rotate(theta, ax, false, false);
	static_cast<tri*>(sides[3])->rotate(theta, ax, false, false);

	base.rotate(theta, ax, false, false);

	if (local) {
		translate(center, false);
	}

	if (reinit)
		init();
}
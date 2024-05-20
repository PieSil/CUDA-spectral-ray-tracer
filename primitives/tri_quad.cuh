#ifndef SPECTRAL_RT_PROJECT_TRI_QUAD_CUH
#define SPECTRAL_RT_PROJECT_TRI_QUAD_CUH

#include "tri.cuh"

/* a class to represent a quad constituted by two triangles*/
class tri_quad {
public:

	__device__
	tri_quad() {}

	__device__
	tri_quad(const point3& Q, const vec3& u, const vec3& v, material* m, tri** _halves, bool defer_init = false) {

		_halves[0] = new tri(Q, u, v, m, defer_init, CreationMode::VECTORS);
		_halves[1] = new tri(Q+u+v, -u, -v, m, defer_init, CreationMode::VECTORS);
		halves[0] = _halves[0];
		halves[1] = _halves[1];
	}

	__device__
	void init() {
		halves[0]->init();
		halves[1]->init();
	}

	__device__
	const vec3 u() const {
		return halves[0]->v[1] - halves[0]->v[0];
	}

	__device__
	const vec3 v() const {
		return halves[0]->v[2] - halves[0]->v[0];
	}

	__device__
	const vec3 Q() const {
		return halves[0]->v[0];
	}

	__device__
	const point3 center() const {
		return ((u() + v()) / 2.0f) + Q();
	}

	__device__
	void flip_normals() {
		halves[0]->flip_normals();
		halves[1]->flip_normals();
	}

	__device__
	void translate(const vec3 dir, const bool reinit = true);

	__device__
	void rotate(const float theta, const transform::AXIS ax, const bool reinit = true, const bool local = true);

	tri* halves[2]; //the tris where the two faces of the quad are stored in
};

#endif 
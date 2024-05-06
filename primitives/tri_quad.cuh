#ifndef SPECTRAL_RT_PROJECT_TRI_QUAD_CUH
#define SPECTRAL_RT_PROJECT_TRI_QUAD_CUH

#include "tri.cuh"

/* a class to represent a quad constituted by two triangles*/
class tri_quad {
public:

	__device__
	tri_quad(const point3& Q, const vec3& u, const vec3& v, material* m, tri** _halves) {

		_halves[0] = new tri(Q, u, v, m, CreationMode::VECTORS);
		_halves[1] = new tri(Q+u+v, -u, -v, m, CreationMode::VECTORS);
		halves = _halves;
	}

	__device__
	vec3 u() {
		return halves[0]->v[1] - halves[0]->v[0];
	}

	__device__
	vec3 v() {
		return halves[0]->v[2] - halves[0]->v[0];
	}

	__device__
	vec3 Q() {
		return halves[0]->v[0];
	}

	__device__
	point3 center() {
		return (u() + v() / 2.0f) + Q();
	}

	tri** halves; //the tris where the two faces of the quad are stored in
};

#endif 
#ifndef SPECTRAL_RT_PROJECT_PYRAMID_CUH
#define SPECTRAL_RT_PROJECT_PYRAMID_CUH

#include "tri_quad.cuh"

class pyramid {
public:
	__device__
	pyramid(point3 Q, vec3 u, vec3 v, vec3 w, material* mats[5], tri** tris, const bool defer_init = false) {
		base = tri_quad(Q, u, v, mats[0], &tris[0], defer_init);
		point3 top = base_center() + w;
		point3 v0 = Q;
		point3 v1 = Q + u;
		point3 v2 = Q + v;
		point3 v3 = v2 + u;

		tris[2] = new tri(v0, top, v2, mats[1], defer_init, CreationMode::VERTICES);
		tris[3] = new tri(v1, top, v0, mats[2], defer_init, CreationMode::VERTICES);
		tris[4] = new tri(v2, top, v3, mats[3], defer_init, CreationMode::VERTICES);
		tris[5] = new tri(v3, top, v1, mats[4], defer_init, CreationMode::VERTICES);

		sides[0] = tris[2];
		sides[1] = tris[3];
		sides[2] = tris[4];
		sides[3] = tris[5];

	}

	__device__
	pyramid(point3 Q, vec3 u, vec3 v, vec3 w, material* m, tri** tris, const bool defer_init = false) {
		base = tri_quad(Q, u, v, m, &tris[0], defer_init);
		point3 top = base_center() + w;
		point3 v0 = Q;
		point3 v1 = Q + u;
		point3 v2 = Q + v;
		point3 v3 = v2 + u;

		tris[2] = new tri(Q, top, v2, m, defer_init, CreationMode::VERTICES);
		tris[3] = new tri(v1, top, Q, m, defer_init, CreationMode::VERTICES);
		tris[4] = new tri(v2, top, v3, m, defer_init, CreationMode::VERTICES);
		tris[5] = new tri(v3, top, v1, m, defer_init, CreationMode::VERTICES);

		sides[0] = tris[2];
		sides[1] = tris[3];
		sides[2] = tris[4];
		sides[3] = tris[5];

	}


	__device__
	const point3 base_center() const {
		return base.center();
	}

	__device__
	void flip_normals() {
		sides[0]->flip_normals();
		sides[1]->flip_normals();
		sides[2]->flip_normals();
		sides[3]->flip_normals();

		base.flip_normals();
	}


	__device__
	void translate(const vec3 dir, const bool reinit = true);

	__device__
	void rotate(const float theta, const transform::AXIS ax, const bool reinit = true, const bool local = true);

private:
	__device__
	void init() {
		sides[0]->init();
		sides[1]->init();
		sides[2]->init();
		sides[3]->init();

		base.init();
	}

	tri_quad base;
	tri* sides[4];
};

#endif
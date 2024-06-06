#ifndef SPECTRAL_RT_PROJECT_PRISM_CUH
#define SPECTRAL_RT_PROJECT_PRISM_CUH

#include "tri.cuh"
#include "tri_quad.cuh"

class prism {
public:

	__device__
		prism(point3 Q, vec3 u, vec3 v, vec3 w, const uint mat_idxs[5], tri** tris, const bool defer_init = false) {
			tris[0] = new tri(Q, v, u, mat_idxs[0], defer_init, CreationMode::VECTORS); //u and v reverted in order to have outward normal
			tris[1] = new tri(Q + w, u, v, mat_idxs[1], defer_init, CreationMode::VECTORS);
			base[0] = tris[0];
			base[1] = tris[1];

			sides[0] = tri_quad(Q, u, w, mat_idxs[2], &tris[2], defer_init);
			sides[1] = tri_quad(Q, w, v, mat_idxs[3], &tris[4], defer_init);
			sides[2] = tri_quad(Q + u, v - u, w, mat_idxs[4], &tris[6], defer_init);
	}

	__device__
		prism(point3 Q, vec3 u, vec3 v, vec3 w, const uint mat_index, tri** tris,  const bool defer_init = false) {
			tris[0] = new tri(Q, v, u, mat_index, defer_init, CreationMode::VECTORS); //bottom
			tris[1] = new tri(Q + w, u, v, mat_index, defer_init, CreationMode::VECTORS); //top
			base[0] = tris[0];
			base[1] = tris[1];

			sides[0] = tri_quad(Q, u, w, mat_index, &tris[2], defer_init);
			sides[1] = tri_quad(Q, w, v, mat_index, &tris[4], defer_init);
			sides[2] = tri_quad(Q + u, v - u, w, mat_index, &tris[6], defer_init);
	}

	__device__
	void init() {
		base[0]->init();
		base[1]->init();

		sides[0].init();
		sides[1].init();
		sides[2].init();
	}

	__device__
	const point3 centroid() {
		point3 v1 = base[0]->v[0];
		point3 v2 = base[0]->v[1];
		point3 v3 = base[0]->v[2];

		point3 v4 = base[1]->v[0];
		point3 v5 = base[1]->v[1];
		point3 v6 = base[1]->v[2];

		return (v1 + v2 + v3 + v4 + v5 + v6) / 6.f;
	}

	__device__
	void flip_normals() {
		base[0]->flip_normals();
		base[1]->flip_normals();

		sides[0].flip_normals();
		sides[1].flip_normals();
		sides[2].flip_normals();
	}

	__device__
	void translate(const vec3 dir, const bool reinit = true);

	__device__
	void rotate(const float theta, const transform::AXIS ax, const bool reinit = true, const bool local = true);

	tri* base[2];
	tri_quad sides[3];
};

#endif
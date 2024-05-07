#ifndef SPECTRAL_RT_PROJECT_PRISM_CUH
#define SPECTRAL_RT_PROJECT_PRISM_CUH

#include "tri.cuh"
#include "tri_quad.cuh"

class prism {
public:

	__device__
		prism(point3 Q, vec3 u, vec3 v, vec3 w, material* mats[5], hittable** tris, const bool defer_init = false) {
			tris[0] = new tri(Q, v, u, mats[0], defer_init, CreationMode::VECTORS); //u and v reverted in order to have outward normal
			tris[1] = new tri(Q + w, u, v, mats[1], defer_init, CreationMode::VECTORS);
			base[0] = tris[0];
			base[1] = tris[1];

			sides[0] = tri_quad(Q, u, w, mats[2], &tris[2], defer_init);
			sides[1] = tri_quad(Q, w, v, mats[3], &tris[4], defer_init);
			sides[2] = tri_quad(Q + u, v - u, w, mats[4], &tris[6], defer_init);
	}

	__device__
		prism(point3 Q, vec3 u, vec3 v, vec3 w, material* m, hittable** tris,  const bool defer_init = false) {
			tris[0] = new tri(Q, v, u, m, defer_init, CreationMode::VECTORS); //bottom
			tris[1] = new tri(Q + w, u, v, m, defer_init, CreationMode::VECTORS); //top
			base[0] = tris[0];
			base[1] = tris[1];

			sides[0] = tri_quad(Q, u, w, m, &tris[2], defer_init);
			sides[1] = tri_quad(Q, w, v, m, &tris[4], defer_init);
			sides[2] = tri_quad(Q + u, v - u, w, m, &tris[6], defer_init);
	}

	__device__
	void init() {
		static_cast<tri*>(base[0])->init();
		static_cast<tri*>(base[1])->init();

		sides[0].init();
		sides[1].init();
		sides[2].init();
	}

	__device__
	const point3 centroid() {
		point3 v1 = static_cast<tri*>(base[0])->v[0];
		point3 v2 = static_cast<tri*>(base[0])->v[1];
		point3 v3 = static_cast<tri*>(base[0])->v[2];

		point3 v4 = static_cast<tri*>(base[1])->v[0];
		point3 v5 = static_cast<tri*>(base[1])->v[1];
		point3 v6 = static_cast<tri*>(base[1])->v[2];

		return (v1 + v2 + v3 + v4 + v5 + v6) / 6.f;
	}

	__device__
	void flip_normals() {
		static_cast<tri*>(base[0])->flip_normals();
		static_cast<tri*>(base[1])->flip_normals();

		sides[0].flip_normals();
		sides[1].flip_normals();
		sides[2].flip_normals();
	}

	__device__
	void translate(const vec3 dir, const bool reinit = true);

	__device__
	void rotate(const float theta, const transform::AXIS ax, const bool reinit = true, const bool local = true);

	hittable* base[2];
	tri_quad sides[3];
};

#endif
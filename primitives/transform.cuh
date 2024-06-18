#ifndef SPECTRAL_RT_PROJECT_TRANSFORM_CUH
#define SPECTRAL_RT_PROJECT_TRANSFORM_CUH

namespace transform {
	enum AXIS {
		NONE,
		X,
		Y,
		Z
	};

	__host__ __device__
	void assign_rot_matrix(float theta, AXIS ax, float rot_matrix[9]);
}

#endif
#include "sellmeier.cuh"

/**
 * @brief Compute Sellmeier's equation using three terms.
 *
 * @param B Array of size 3 containg B coefficients.
 * @param C Array of size 3 containg C coefficients.
 * @param lambda The wavelength to compute sellmeier's equation for (in nm).
 * @return The refractive index at the given wavelength.
 */
__device__
float sellmeier_index(const float b[3], const float c[3], float lambda) {
	lambda *= 1e-3f; //convert to micrometers
	float lambda_squared = lambda * lambda;

	float index = 1.0f +
		(b[0] * lambda_squared) / (lambda_squared - c[0]) +
		(b[1] * lambda_squared) / (lambda_squared - c[1]) +
		(b[2] * lambda_squared) / (lambda_squared - c[2]);

	index = sqrtf(index);
	return index;
}
//
// Created by pietr on 31/03/2024.
//

#include "color.cuh"

__host__ __device__
float invert_channel_correction(const float value) {
    //sRGB inverse gamma correction

    float res = value < 0.04045 ? value/12.92f : pow(((value+0.055f)/1.055f), 2.4f);
    return res;
}

__host__ __device__
float correct_channel(const float value) {
    //sRGB gamma correction

    float res = value < 0.0f ? 0.0f : (value < 0.0031308f ? 12.92f*value : ( value < 1.0f ? ((1.055f*pow(value, 0.416666f)) - 0.055f) : 1.0f));

    return res;
}

__host__ __device__
color sRGB_to_XYZ(const color sRGB, const float *sRGB_to_XYZ_matrix) {
    color inv_corr_srgb = color(invert_channel_correction(sRGB.x()),
                           invert_channel_correction(sRGB.y()),
                           invert_channel_correction(sRGB.z()));

    color xyz = color::matrix_mul(inv_corr_srgb, sRGB_to_XYZ_matrix);

    return xyz;
}

__host__ __device__
color XYZ_to_sRGB(const color xyz, const float *XYZ_to_sRGB_matrix) {

    color sRGB_not_corrected = color::matrix_mul(xyz, XYZ_to_sRGB_matrix);

    return {correct_channel(sRGB_not_corrected.x()), correct_channel(sRGB_not_corrected.y()), correct_channel(sRGB_not_corrected.z())};
}

__host__ __device__
color expand_sRGB(const color bounded_sRGB) {
    // expand component of color from [0, 1] to [0, 255]
    return {float(int(bounded_sRGB[0]*255.99f)),
            float(int(bounded_sRGB[1]*255.99f)),
            float(int(bounded_sRGB[2]*255.99f))};
}


/**
 * @brief Converts a sampled spectrum to an XYZ color given its power distribution.
 *
 * @param spectrum the wavelengths which define the sampled spectrum.
 * @param power_distribution the power distribution of the sampled spectrum, one vale for each wavelength.
 * @param n_samples the number of samples of the spectrum.
 * @return the approximated XYZ color resulting from the sampled spectrum.
 */
__host__
color spectrum_to_XYZ(const float* spectrum, const float* power_distribution, int n_samples) {
    float delta_lambda = (LAMBDA_MAX - LAMBDA_MIN) / float(n_samples);
    float x = 0.0f, y = 0.0f, z = 0.0f;

    for (int i = 0; i < n_samples; i++) {
        float lambda = spectrum[i];
        float power = power_distribution[i];
        x += spectrum_interp(cie_x, lambda) * power * delta_lambda;
        y += spectrum_interp(cie_y, lambda) * power * delta_lambda;
        z += spectrum_interp(cie_z, lambda) * power * delta_lambda;
    }

    //color xyz = color(x, y, z);

    return color(x, y, z);
}

/**
 * @brief Device function. Converts a sampled spectrum to an XYZ color given its power distribution.
 *
 * The device version of spectrum_to_XYZ()
 *
 * @param spectrum the wavelengths which define the sampled spectrum.
 * @param power_distribution the power distribution of the sampled spectrum, one vale for each wavelength.
 * @param n_samples the number of samples of the spectrum.
 * @return the approximated XYZ color resulting from the sampled spectrum.
 */
__device__
color dev_spectrum_to_XYZ(float* spectrum, float* power_distribution, int n_samples) {
    float delta_lambda = (LAMBDA_MAX - LAMBDA_MIN) / float(n_samples);
    float x = 0.0f, y = 0.0f, z = 0.0f;

    for (int i = 0; i < n_samples; i++) {
        float lambda = spectrum[i];
        float power = power_distribution[i];
        x += spectrum_interp(dev_cie_x, lambda) * power * delta_lambda;
        y += spectrum_interp(dev_cie_y, lambda) * power * delta_lambda;
        z += spectrum_interp(dev_cie_z, lambda) * power * delta_lambda;
    }

    // color xyz = color(x, y, z);

    return color(x, y, z);
}

__host__
void write_color(std::ostream &out, const color pixel_color) {
    // Assumes RGB values are already in [0, 255] range

    out << static_cast<int>(pixel_color.x()) << ' '
        << static_cast<int>(pixel_color.y()) << ' '
        << static_cast<int>(pixel_color.z()) << '\n';
}
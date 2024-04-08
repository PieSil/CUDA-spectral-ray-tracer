//
// Created by pietr on 03/04/2024.
//

#ifndef SPECTRAL_RT_PROJECT_DEVICE_INIT_CUH
#define SPECTRAL_RT_PROJECT_DEVICE_INIT_CUH

#include "cie_const.cuh"
#include "color_const.cuh"
#include "srgb_to_spectrum.h"

__constant__ inline int dev_sRGBToSpectrumTable_Res;
__constant__ inline float dev_sRGBToSpectrumTable_Scale[64];

__host__
inline void init_device_symbols() {
    //init device constant memory

    /* sRGB to XYZ (and back) conversion matrices */
    checkCudaErrors(cudaMemcpyToSymbol(dev_d65_XYZ_to_sRGB, d65_XYZ_to_sRGB, 3 * 3 * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(dev_d65_sRGB_to_XYZ, d65_sRGB_to_XYZ, 3 * 3 * sizeof(float)));

    /* cie color response curves */
    checkCudaErrors(cudaMemcpyToSymbol(dev_cie_x, cie_x, N_CIE_SAMPLES*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(dev_cie_y, cie_y, N_CIE_SAMPLES*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(dev_cie_z, cie_z, N_CIE_SAMPLES*sizeof(float)));

    /*
     * color to spectrum conversion
     * NOTE: Data table is omitted since it's too big for constant memory
     */
    checkCudaErrors(cudaMemcpyToSymbol(dev_sRGBToSpectrumTable_Scale, sRGBToSpectrumTable_Scale, 64*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(dev_sRGBToSpectrumTable_Res, &sRGBToSpectrumTable_Res, sizeof(int)));
}

#endif //SPECTRAL_RT_PROJECT_DEVICE_INIT_CUH

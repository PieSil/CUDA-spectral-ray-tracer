//
// Created by pietr on 23/11/2023.
//

#ifndef RTWEEKEND_CUDA_CUDA_UTILITY_CUH
#define RTWEEKEND_CUDA_CUDA_UTILITY_CUH

#include <curand_kernel.h>
#include <iostream>
#include <string>

typedef unsigned int uint;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
__host__
void check_cuda(cudaError_t result, std::string func, std::string file, int line);

__device__
float cuda_random_float(curandState* local_rand_state);

__device__
float cuda_random_float(float min, float max, curandState* local_rand_state);

__device__
int cuda_random_int(int min, int max, curandState* local_rand_state);

__device__
float device_clamp(float value, float min, float max);

__device__
void random_permutation(int *indices, int size, curandState *local_rand_state);

#endif //RTWEEKEND_CUDA_CUDA_UTILITY_CUH

#ifndef RTWEEKEND_CUDA_CUDA_UTILITY_CUH
#define RTWEEKEND_CUDA_CUDA_UTILITY_CUH

//IntelliSense hack
#ifndef __CUDACC__
#include <device_launch_parameters.h>
#endif

#include "utility.h"
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

__device__
inline bool thread_matches(uint tx = 0, uint ty = 0, uint bx = 0, uint by = 0) {
    return (threadIdx.x == tx && threadIdx.y == ty && blockIdx.x == bx && blockIdx.y == by);
}

__host__ __device__
inline float degrees_to_radians(float degrees) {
    return degrees * PI / 180.0f;
}

#endif //RTWEEKEND_CUDA_CUDA_UTILITY_CUH

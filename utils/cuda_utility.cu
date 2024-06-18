//
// Created by pietr on 24/11/2023.
//

#include "cuda_utility.cuh"

__host__
void check_cuda(cudaError_t result, const std::string func, const std::string file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        std::cerr << "Error msg: " << cudaGetErrorString(result) << "\n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__
float cuda_random_float(curandState* local_rand_state) {
    // Returns a random real in (0,1].
    float random = curand_uniform(local_rand_state);
    return random;
}

__device__
float cuda_random_float(float min, float max, curandState* local_rand_state) {
    // Returns a random real in (min,max].

    float range_width = max - min;
    float random = cuda_random_float(local_rand_state); // Generates random float between 0 and 1
    float result = random * range_width + min; // Scale and shift to desired range (min, max]

    return result;
}

__device__
int cuda_random_int(int min, int max, curandState* local_rand_state) {
    //returns a random int in [min, max]
    float random_float = cuda_random_float(static_cast<float>(min-1), static_cast<float>(max-1), local_rand_state);
    return static_cast<int>(ceil(random_float));
}

__device__
float device_clamp(float value, float min, float max) {
    //Do not return early, avoid divergence, maybe I'm paranoid

    const float t = value < min ? min : value;
    return t > max ? max : t;
}

__device__
void random_permutation(int *indices, int size, curandState *local_rand_state) {
    //Shuffle an array of indices
    for (int i = 0; i < size; ++i) {
        indices[i] = i;
    }

    for (int i = size - 1; i > 0; --i) {
        int j = cuda_random_int(0, i, local_rand_state);

        // Swap indices[i] with indices[j]
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

}
#include <iostream>
#include "vec3.cuh"
#include "device_init.cuh"
#include "color_to_spectrum.cuh"
#include "hittable.cuh"
#include "materials/material.cuh"
#include "bvh.cuh"
#include "rendering.cuh"
#include "camera_builder.cuh"

#define SAMPLES_PER_PIXEL 500
#define BOUNCE_LIMIT 10

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}

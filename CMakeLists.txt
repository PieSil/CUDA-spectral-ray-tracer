cmake_minimum_required(VERSION 3.24)

project(spectral-rt-project CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(INCLUDE_DIRS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES")
include_directories(INCLUDE_DIRS)

set(CMAKE_CUDA_STANDARD 17)

# set_target_properties(srgb_to_spectrum_pch PROPERTIES LINKER_LANGUAGE CXX)

add_executable(spectral-rt-project
        main.cu
        cie_const.cu
        spectrum.cu
        color_const.cu
        cuda_utility.cu
        color.cu
        srgb_to_spectrum.cu
        sphere.cu
        materials/material.cu
        aabb.cu
        bvh.cu
        rendering.cu
        camera.cu
        world.cu)

set_target_properties(spectral-rt-project PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "50;75;80"
        CXX_PRECOMPILE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/srgb_to_spectrum.cuh)

cmake_minimum_required(VERSION 3.20)
enable_language(CUDA)

project(spectral-rt-project CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(SUBDIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/materials")

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
        scene.cu
        quad.cu
        transform.cu
        sellmeier.cu
        tri.cu)

set_target_properties(spectral-rt-project PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "50;75;80"
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CXX_PRECOMPILE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/srgb_to_spectrum.cuh)

target_include_directories(spectral-rt-project PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SUBDIRECTORIES}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CUDA_INCLUDE_DIRS})

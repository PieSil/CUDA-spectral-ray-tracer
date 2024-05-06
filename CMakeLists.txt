cmake_minimum_required(VERSION 3.20)
enable_language(CUDA)

project(spectral-rt-project CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(SUBDIRECTORIES 
    "${CMAKE_CURRENT_SOURCE_DIR}/materials"
    "${CMAKE_CURRENT_SOURCE_DIR}/primitives"
    "${CMAKE_CURRENT_SOURCE_DIR}/bvh"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils"
    "${CMAKE_CURRENT_SOURCE_DIR}/rendering"
    "${CMAKE_CURRENT_SOURCE_DIR}/refraction"
    "${CMAKE_CURRENT_SOURCE_DIR}/color"
    "${CMAKE_CURRENT_SOURCE_DIR}/spectrum"
    "${CMAKE_CURRENT_SOURCE_DIR}/ray"
    "${CMAKE_CURRENT_SOURCE_DIR}/math"
    "${CMAKE_CURRENT_SOURCE_DIR}/io"
    "${CMAKE_CURRENT_SOURCE_DIR}/scene")

add_executable(spectral-rt-project
        main.cu
        utils/cie_const.cu
        spectrum/spectrum.cu
        utils/color_const.cu
        utils/cuda_utility.cu
        color/color.cu
        utils/srgb_to_spectrum.cu
        primitives/sphere.cu
        materials/material.cu
        bvh/aabb.cu
        bvh/bvh.cu
        rendering/rendering.cu
        rendering/camera.cu
        scene/scene.cu
        primitives/quad.cu
        primitives/transform.cu
        refraction/sellmeier.cu
        primitives/tri.cu
        primitives/tri_quad.cu)

set_target_properties(spectral-rt-project PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "50;75;80"
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CXX_PRECOMPILE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/utils/srgb_to_spectrum.cuh)

target_include_directories(spectral-rt-project PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SUBDIRECTORIES}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CUDA_INCLUDE_DIRS})

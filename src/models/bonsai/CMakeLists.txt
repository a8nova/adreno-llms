cmake_minimum_required(VERSION 3.18)

# CMake 4.x compat
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27")
  cmake_policy(SET CMP0148 OLD)
endif()
if(DEFINED CMAKE_POLICY_VERSION_MINIMUM)
  # Already set
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
endif()

project(bonsai CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Release: -O3 -ffast-math (matches the token-exact hand-build; -ffast-math is
# part of the frozen numeric configuration). Debug: -O0 -g.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 16 KB page alignment for Google Play (Android 15 / API 35+). NDK r26 emits
# 4 KB LOAD segments by default; force 0x4000 on BOTH the exe and any shared
# lib. Segment alignment only — behavior-neutral.
if(ANDROID)
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,-z,max-page-size=16384")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,-z,max-page-size=16384")
endif()

# Architecture hint for Android NDK toolchain
if(ANDROID)
    set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
endif()

# ──────────────────────────────────────────────────────────────────────
# BONSAI_FP16_STORAGE: storage precision for ACTIVATIONS on device.
# fp32 default (the token-exact gate configuration; weights are always 1-bit
# packed regardless). Pass -DBONSAI_FP16_STORAGE=1 to flip to half — the only
# thing that changes is the binary suffix (_f16) and the on-device storage_t.
# ──────────────────────────────────────────────────────────────────────
if(BONSAI_FP16_STORAGE)
    add_compile_definitions(BONSAI_FP16_STORAGE=1)
    set(BONSAI_BIN_SUFFIX "_f16")
    message(STATUS "BONSAI_FP16_STORAGE=1 — binary suffix '_f16'")
else()
    set(BONSAI_BIN_SUFFIX "")
    message(STATUS "BONSAI storage fp32 (default)")
endif()

# ──────────────────────────────────────────────────────────────────────
# BONSAI_HOST_ONLY: build the CPU reference forward with NO OpenCL at all.
# This is the development loop for a NEW ARCHITECTURE: reference_model.h is
# self-contained (mmap'd .nnb + q1_row_decode + fp32 gemv), so the math can be
# iterated on the dev machine in seconds and diffed against the llama.cpp
# oracle, before a single kernel is written. Device correctness is then a
# translation problem with a known-good target rather than a search.
#   ./scripts/build_host.sh   ->  build/host/bonsai_ref
# ──────────────────────────────────────────────────────────────────────
if(BONSAI_HOST_ONLY)
    message(STATUS "BONSAI_HOST_ONLY=1 — CPU reference only, no OpenCL")
else()

# OpenCL headers/library supplied by the porting orchestrator / build.sh via -D.
if(DEFINED OPENCL_INCLUDE_DIR)
    message(STATUS "Using OpenCL headers from: ${OPENCL_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "OPENCL_INCLUDE_DIR not set. Ensure build.sh / orchestrator passes it.")
endif()

# Our Khronos headers take priority over any system OpenCL path.
include_directories(BEFORE ${OPENCL_INCLUDE_DIR})

if(DEFINED OPENCL_LIBRARIES)
    message(STATUS "Using provided OpenCL library: ${OPENCL_LIBRARIES}")
elseif(ANDROID)
    message(FATAL_ERROR "OPENCL_LIBRARIES not provided. Extract libOpenCL.so from the device.")
else()
    if(APPLE)
        set(OPENCL_LIBRARIES "-framework OpenCL")
    else()
        set(OPENCL_LIBRARIES "OpenCL")
    endif()
endif()

# Bonsai is BONSAI_OPENCL-only (the device model). CL 1.2 target.
add_compile_definitions(BONSAI_OPENCL=1 CL_TARGET_OPENCL_VERSION=120)

endif()

# Layer sources (out-of-line DeviceModel method bodies).
set(LAYER_SOURCES
    src/layers/attention.cpp
    src/layers/embedding.cpp
    src/layers/layer_norm.cpp
    src/layers/mlp.cpp
)

# Source files (test_q1.cpp is a standalone unit test — NOT in the build).
if(BONSAI_HOST_ONLY)
    # No model.cpp / opencl_context.cpp / layers — those ARE the device model.
    # prof.cpp is excluded too: it profiles CL events (includes <CL/cl.h>) and
    # main.cpp never references it.
    set(SOURCES src/main.cpp src/tokenizer.cpp)
else()
    set(SOURCES
        src/main.cpp
        src/model.cpp
        src/tokenizer.cpp
        src/opencl_context.cpp
        src/prof.cpp
        ${LAYER_SOURCES}
    )
endif()

add_executable(bonsai_inference ${SOURCES})
target_include_directories(bonsai_inference PRIVATE src/ src/layers/)
set_target_properties(bonsai_inference PROPERTIES
    OUTPUT_NAME "bonsai_inference${BONSAI_BIN_SUFFIX}"
)

# Link OpenCL (+ liblog on Android). Host-only build links nothing extra.
if(BONSAI_HOST_ONLY)
    set_target_properties(bonsai_inference PROPERTIES OUTPUT_NAME "bonsai_ref")
else()
    target_link_libraries(bonsai_inference ${OPENCL_LIBRARIES})
endif()

if(ANDROID)
    target_link_libraries(bonsai_inference log)
    target_link_options(bonsai_inference PRIVATE
        "-Wl,--allow-shlib-undefined"
        "-Wl,--unresolved-symbols=ignore-all"
        "-Wl,-rpath,/vendor/lib64:/vendor/lib:/system/lib64:/system/lib"
    )
    set_target_properties(bonsai_inference PROPERTIES
        INSTALL_RPATH "/vendor/lib64:/vendor/lib:/system/lib64:/system/lib"
    )
    message(STATUS "Configured runtime RPATH for OpenCL on Android device")
endif()

# Copy kernel files next to the binary so it can find them at "kernels/".
file(GLOB KERNEL_FILES "${CMAKE_SOURCE_DIR}/kernels/*.cl")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/kernels")
foreach(KERNEL_FILE ${KERNEL_FILES})
    get_filename_component(KERNEL_NAME ${KERNEL_FILE} NAME)
    configure_file(${KERNEL_FILE} "${CMAKE_BINARY_DIR}/kernels/${KERNEL_NAME}" COPYONLY)
endforeach()

install(TARGETS bonsai_inference DESTINATION bin)

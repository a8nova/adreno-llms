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

project(bonsai27b CXX)

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

# ── CLBlast: dense GEMM for the VISION tower only ────────────────────────────
# The text tower needs none of this — its weights are 1-bit and q1_gemv/q1_gemm handle them. The
# Qwen3-VL vision tower is dense Q8_0/F16, ~1.6 TMAC per image, and needs a real GEMM. CLBlast is
# already proven on Adreno by the sibling ports (lfm2-5-350m, granite, depth-anything-v2-small) and
# Edgi already ships a 16 KB-aligned libclblast.so, so the runtime side is solved.
#
# Gated on BONSAI_VISION so a text-only build neither fetches nor links it.
option(BONSAI_VISION "Build the vision tower (needs CLBlast)" OFF)
if(BONSAI_VISION)
    # CLBlast runs its OWN FindOpenCL, which does not know about our OPENCL_INCLUDE_DIR /
    # OPENCL_LIBRARIES (we point at an extracted-from-device libOpenCL.so, not a system SDK).
    # Seed the cache variables its find_package looks for, exactly as the sibling ports do,
    # or the configure dies with "Could NOT find OpenCL (missing: OPENCL_INCLUDE_DIRS)".
    set(OpenCL_INCLUDE_DIR  "${OPENCL_INCLUDE_DIR}" CACHE PATH   "OpenCL include dir"  FORCE)
    set(OpenCL_INCLUDE_DIRS "${OPENCL_INCLUDE_DIR}" CACHE PATH   "OpenCL include dirs" FORCE)
    set(OpenCL_LIBRARY      "${OPENCL_LIBRARIES}"   CACHE STRING "OpenCL library"      FORCE)
    set(OpenCL_LIBRARIES    "${OPENCL_LIBRARIES}"   CACHE STRING "OpenCL libraries"    FORCE)
    set(OPENCL_ROOT         "${OPENCL_INCLUDE_DIR}/.." CACHE PATH "OpenCL root"        FORCE)
    # The names CLBlast's bundled FindOpenCL.cmake actually tests are UPPERCASE, and under the
    # Android toolchain its find_path() only searches the NDK sysroot (CMAKE_FIND_ROOT_PATH_MODE
    # _INCLUDE=ONLY), so it can never see our extracted-from-device headers. Pre-setting the cache
    # variables makes find_path a no-op and the find_package succeeds with our paths.
    set(OPENCL_INCLUDE_DIRS "${OPENCL_INCLUDE_DIR}" CACHE PATH   "OpenCL include dirs" FORCE)
    set(OPENCL_LIBRARIES    "${OPENCL_LIBRARIES}"   CACHE STRING "OpenCL libraries"    FORCE)
    set(NNOPT_CLBLAST_DIR "$ENV{HOME}/.nnopt/deps/clblast")
    if(EXISTS "${NNOPT_CLBLAST_DIR}/lib/libclblast.a" AND EXISTS "${NNOPT_CLBLAST_DIR}/include/clblast.h")
        message(STATUS "Using pre-built CLBlast from ${NNOPT_CLBLAST_DIR}")
        add_library(clblast STATIC IMPORTED GLOBAL)
        set_target_properties(clblast PROPERTIES
            IMPORTED_LOCATION "${NNOPT_CLBLAST_DIR}/lib/libclblast.a"
            INTERFACE_INCLUDE_DIRECTORIES "${NNOPT_CLBLAST_DIR}/include")
        include_directories("${NNOPT_CLBLAST_DIR}/include")
    else()
        message(STATUS "Pre-built CLBlast not found — fetching from source (first build only)...")
        include(FetchContent)
        FetchContent_Declare(clblast
            GIT_REPOSITORY https://github.com/CNugteren/CLBlast.git
            GIT_TAG 1.6.3 GIT_SHALLOW TRUE)
        # Android: shared, deployed alongside the binary (matches the sibling ports and Edgi's
        # jniLibs). Host: static, to avoid loader-path concerns.
        # STATIC on Android, unlike the sibling ports which ship a shared libclblast.so.
        # Shared linking put a libc++ symbol resolution through libclblast.so instead of into this
        # binary's c++_static runtime, and the device then failed at exec with
        #   cannot locate symbol _ZNKSt6__ndk120__vector_base_commonILb1EE20__throw_length_errorEv
        # because the libclblast.so actually present was built with different STL settings. Static
        # makes libbonsai-27b.so self-contained: no external CLBlast, no STL mismatch to get wrong.
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "CLBlast static — self-contained binary" FORCE)
        set(SAMPLES OFF CACHE BOOL "" FORCE)
        set(TUNERS OFF CACHE BOOL "" FORCE)
        set(CLIENTS OFF CACHE BOOL "" FORCE)
        set(TESTS OFF CACHE BOOL "" FORCE)
        set(NETLIB OFF CACHE BOOL "" FORCE)
        # HALF=ON so clblast::Gemm<half> (Hgemm) exists. The vision tower MUST be fp16 on device:
        # dequantised to fp32 its 461M params are 1.84 GB, which cannot coexist with the 3617 MB
        # text tower in a 5542 MB budget. At fp16 it is 922 MB, and it is freed after the image is
        # encoded. Without HALF=ON the Hgemm symbol is absent and the call silently falls back to
        # Sgemm, which would defeat the point.
        set(HALF ON CACHE BOOL "CLBlast fp16 routines" FORCE)
        FetchContent_MakeAvailable(clblast)
        if(ANDROID AND TARGET clblast)
            target_compile_options(clblast PRIVATE "-include" "sstream")
        endif()
        include_directories(${clblast_SOURCE_DIR}/include)
    endif()
    add_compile_definitions(BONSAI_VISION=1)
endif()

# stb_image — single-header decode for the VLM path. Header-only; the matching
# src/third_party/stb_image_impl.cpp is the one TU that defines the implementation.
# Fetched only for vision builds; a text-only build never sees these symbols.
if(BONSAI_VISION)
    include(FetchContent)
    FetchContent_Declare(stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(stb)
    include_directories(${stb_SOURCE_DIR})
endif()

# Bonsai is BONSAI_OPENCL-only (the device model). CL 1.2 target.
add_compile_definitions(BONSAI_OPENCL=1 CL_TARGET_OPENCL_VERSION=120)

endif()

# Layer sources (out-of-line DeviceModel method bodies).
set(LAYER_SOURCES
    src/layers/common.cpp
    src/layers/linear_attn.cpp
    src/layers/full_attn.cpp
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
        $<$<BOOL:${BONSAI_VISION}>:src/vision_model.cpp>
        $<$<BOOL:${BONSAI_VISION}>:src/third_party/stb_image_impl.cpp>
    )
endif()

add_executable(bonsai_inference ${SOURCES})
target_include_directories(bonsai_inference PRIVATE src/ src/layers/)
set_target_properties(bonsai_inference PROPERTIES
    OUTPUT_NAME "bonsai27b_inference${BONSAI_BIN_SUFFIX}"
)

# Link OpenCL (+ liblog on Android). Host-only build links nothing extra.
if(BONSAI_HOST_ONLY)
    set_target_properties(bonsai_inference PROPERTIES OUTPUT_NAME "bonsai_ref")
else()
    target_link_libraries(bonsai_inference ${OPENCL_LIBRARIES})
    if(BONSAI_VISION)
        target_link_libraries(bonsai_inference clblast)
    endif()
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

# Standalone device-compiler check for the kernels (no weights needed).
if(NOT BONSAI_HOST_ONLY)
    add_executable(kernel_check src/kernel_check.cpp src/opencl_context.cpp)
    target_include_directories(kernel_check PRIVATE src/ src/layers/)
    target_link_libraries(kernel_check ${OPENCL_LIBRARIES})
    if(ANDROID)
        target_link_libraries(kernel_check log)
        target_link_options(kernel_check PRIVATE
            "-Wl,--allow-shlib-undefined"
            "-Wl,--unresolved-symbols=ignore-all"
            "-Wl,-rpath,/vendor/lib64:/vendor/lib:/system/lib64:/system/lib")
    endif()
endif()

# Numerical differential test for delta_net.cl against gdn::step (device-side).
if(NOT BONSAI_HOST_ONLY)
    add_executable(test_delta_net_cl src/test_delta_net_cl.cpp src/opencl_context.cpp)
    target_include_directories(test_delta_net_cl PRIVATE src/ src/layers/)
    target_link_libraries(test_delta_net_cl ${OPENCL_LIBRARIES})
    if(ANDROID)
        target_link_libraries(test_delta_net_cl log)
        target_link_options(test_delta_net_cl PRIVATE
            "-Wl,--allow-shlib-undefined"
            "-Wl,--unresolved-symbols=ignore-all"
            "-Wl,-rpath,/vendor/lib64:/vendor/lib:/system/lib64:/system/lib")
    endif()
endif()

install(TARGETS bonsai_inference DESTINATION bin)

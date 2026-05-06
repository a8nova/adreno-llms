cmake_minimum_required(VERSION 3.18)

# CMake 4.x compat: allow FetchContent deps with old cmake_minimum_required
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27")
  cmake_policy(SET CMP0148 OLD)
endif()
if(DEFINED CMAKE_POLICY_VERSION_MINIMUM)
  # Already set
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
endif()

project(SmolLM2_135M_Instruct_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Debug mode: NNOPT_DEBUG enables checkpoints, layer checks, GPU memory tracking.
# Pass -DNNOPT_DEBUG=1 from build.sh (default: Debug with macros ON).
if(NNOPT_DEBUG)
    add_definitions(-DNNOPT_DEBUG)
    message(STATUS "NNOPT debug macros ENABLED (checkpoints, layer checks, GPU mem tracking)")
else()
    message(STATUS "NNOPT debug macros DISABLED (release mode — errors only)")
endif()

# ──────────────────────────────────────────────────────────────────────
# NNOPT_DTYPE: storage precision for activations + most weights.
# fp32 default; pass -DNNOPT_DTYPE=fp16 to flip the whole binary to half.
# Same source tree, same scripts, same workflow — only the binary suffix
# (_fp16) and the runtime weight file (weights/model.fp16.bin) change.
# Layer code uses nnopt_storage_t; kernels use storage_t + LOAD/STORE.
# Mechanically enforced by the Build tool (cppStandards Rule 01).
# ──────────────────────────────────────────────────────────────────────
set(NNOPT_DTYPE "fp32" CACHE STRING "Storage dtype: fp32 or fp16")
set_property(CACHE NNOPT_DTYPE PROPERTY STRINGS fp32 fp16)
if(NNOPT_DTYPE STREQUAL "fp16")
    add_compile_definitions(NNOPT_USE_FP16=1)
    set(NNOPT_BIN_SUFFIX "_fp16")
    # CLBlast must be built with HALF=ON for HGemm symbols.
    set(HALF ON CACHE BOOL "Build CLBlast HGemm" FORCE)
    message(STATUS "NNOPT_DTYPE=fp16 — binary suffix '_fp16', CLBlast HALF=ON")
else()
    set(NNOPT_BIN_SUFFIX "")
    message(STATUS "NNOPT_DTYPE=fp32 (default)")
endif()

# Architecture hint for Android NDK toolchain
if(ANDROID)
    set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
endif()

# Generic OpenCL backend
if(DEFINED OPENCL_INCLUDE_DIR)
    message(STATUS "Using OpenCL headers from: ${OPENCL_INCLUDE_DIR}")
    include_directories(BEFORE ${OPENCL_INCLUDE_DIR})
    set(BACKEND_LIBRARIES OpenCL)
    # When cross-compiling, FindOpenCL may only provide headers. The actual
    # libOpenCL.so path is injected by build.sh via OPENCL_LIBRARIES.
    if(DEFINED OPENCL_LIBRARIES)
      set(BACKEND_LIBRARIES ${OPENCL_LIBRARIES})
    endif()
else()
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        include_directories(${OpenCL_INCLUDE_DIRS})
        set(BACKEND_LIBRARIES ${OpenCL_LIBRARIES})
        message(STATUS "Using system OpenCL backend")
    else()
        message(WARNING "OpenCL not found. Build may fail. Run build.sh to auto-download deps.")
        set(BACKEND_LIBRARIES "")
    endif()
endif()


# Layer sources — explicitly listed
set(LAYER_SOURCES
    src/layers/attention.cpp
    src/layers/embedding.cpp
    src/layers/layer_norm.cpp
    src/layers/lm_head.cpp
    src/layers/mlp.cpp
)

# Source files
set(SOURCES
    src/main.cpp
    src/model.cpp
    src/sampler.cpp
    src/opencl_context.cpp
    src/weights.cpp
    src/tokenizer.cpp
    src/utils.cpp
    src/prof.cpp
    ${LAYER_SOURCES}
)

# Main inference binary
add_executable(SmolLM2_135M_Instruct_inference ${SOURCES})
target_include_directories(SmolLM2_135M_Instruct_inference PRIVATE src/ src/layers/)

# CLBlast (FetchContent) — required for pytorch_linear GEMMs.
# Android runtime note:
# - Preferred: static link so we don't need libclblast.so on device.
# - Fallback: if a shared lib still gets produced/linked, deploy_android.sh
#   pushes it and run_android.sh sets LD_LIBRARY_PATH.
include(FetchContent)
FetchContent_Declare(
  clblast
  GIT_REPOSITORY https://github.com/CNugteren/CLBlast.git
  GIT_TAG 1.6.2
)
set(CLBlast_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLBlast_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(CLBlast_BUILD_CLIENTS OFF CACHE BOOL "" FORCE)
set(CLBlast_WITH_CLHPP OFF CACHE BOOL "" FORCE)
# Force static-only.
set(CLBlast_SHARED_LIBRARY OFF CACHE BOOL "" FORCE)
set(CLBlast_STATIC_LIBRARY ON  CACHE BOOL "" FORCE)
# Some CLBlast versions use these option names instead.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CLBlast_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(CLBlast_BUILD_STATIC ON CACHE BOOL "" FORCE)
# Propagate our OpenCL headers path to CLBlast's FindOpenCL.
if(DEFINED OPENCL_INCLUDE_DIR)
  set(OPENCL_INCLUDE_DIRS ${OPENCL_INCLUDE_DIR} CACHE PATH "" FORCE)
endif()
FetchContent_MakeAvailable(clblast)

# Force-link the static archive if available; refuse shared-only targets.
if(TARGET clblast_static)
  target_link_libraries(SmolLM2_135M_Instruct_inference clblast_static)
elseif(TARGET clblast)
  # Some builds expose only 'clblast' even when configured static.
  # If it's actually SHARED, we'll detect below.
  target_link_libraries(SmolLM2_135M_Instruct_inference clblast)
else()
  message(FATAL_ERROR "CLBlast target not found (expected clblast_static or clblast).")
endif()

# Fail fast if we accidentally linked against a shared CLBlast target.
foreach(_tgt IN ITEMS clblast clblast_static)
  if(TARGET ${_tgt})
    get_target_property(_tgt_type ${_tgt} TYPE)
    if(_tgt_type STREQUAL "SHARED_LIBRARY")
      message(FATAL_ERROR "CLBlast resolved to a SHARED_LIBRARY target (${_tgt}). This would require libclblast.so on device.")
    endif()
  endif()
endforeach()

target_include_directories(SmolLM2_135M_Instruct_inference PRIVATE ${clblast_SOURCE_DIR}/include)
# Suffix the binary with NNOPT_BIN_SUFFIX so fp32 (<name>_inference) and fp16
# (<name>_inference_fp16) coexist on the device side-by-side.
set_target_properties(SmolLM2_135M_Instruct_inference PROPERTIES
    OUTPUT_NAME "SmolLM2_135M_Instruct_inference${NNOPT_BIN_SUFFIX}"
)

# Link libraries (backend-specific)
target_link_libraries(SmolLM2_135M_Instruct_inference ${BACKEND_LIBRARIES})

# If OPENCL_LIBRARIES is a full path to libOpenCL.so (the Android stub we
# extracted), make sure the linker can locate it.
if(DEFINED OPENCL_LIBRARIES)
  get_filename_component(_opencl_lib_dir "${OPENCL_LIBRARIES}" DIRECTORY)
  if(EXISTS "${OPENCL_LIBRARIES}")
    target_link_directories(SmolLM2_135M_Instruct_inference PRIVATE "${_opencl_lib_dir}")
  endif()
endif()

# Android-specific linker flags for runtime OpenCL resolution
if(ANDROID)
    target_link_options(SmolLM2_135M_Instruct_inference PRIVATE
        "-Wl,--allow-shlib-undefined"
        "-Wl,--unresolved-symbols=ignore-all"
        "-Wl,-rpath,/vendor/lib64:/vendor/lib:/system/lib64:/system/lib"
    )
    set_target_properties(SmolLM2_135M_Instruct_inference PROPERTIES
        INSTALL_RPATH "/vendor/lib64:/vendor/lib:/system/lib64:/system/lib"
    )
    message(STATUS "Configured runtime RPATH for OpenCL on Android device")
endif()

# Copy kernel files to build directory so the binary can find them at "kernels/"
file(GLOB KERNEL_FILES "${CMAKE_SOURCE_DIR}/kernels/*.cl")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/kernels")
foreach(KERNEL_FILE ${KERNEL_FILES})
    get_filename_component(KERNEL_NAME ${KERNEL_FILE} NAME)
    configure_file(${KERNEL_FILE} "${CMAKE_BINARY_DIR}/kernels/${KERNEL_NAME}" COPYONLY)
endforeach()

# Install
install(TARGETS SmolLM2_135M_Instruct_inference DESTINATION bin)

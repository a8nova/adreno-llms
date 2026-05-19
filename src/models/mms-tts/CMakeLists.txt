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

project(mms_tts_eng_inference CXX)

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

# CLBlast / OpenCL backend (backend="opencl")
# OpenCL headers from ~/.nnopt/deps/ (downloaded by porting orchestrator)
# These are Khronos headers with an OpenCL/ symlink for macOS compatibility
if(DEFINED OPENCL_INCLUDE_DIR)
    message(STATUS "Using OpenCL headers from: ${OPENCL_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "OPENCL_INCLUDE_DIR not set. Ensure porting orchestrator has run.")
endif()

# Pre-populate ALL OpenCL find_package result variables BEFORE FetchContent.
# This prevents find_package(OpenCL) from searching and finding macOS framework headers.
# We use Khronos headers for includes, but macOS framework for linking.
set(OpenCL_FOUND TRUE CACHE BOOL "OpenCL found (pre-set)" FORCE)
set(OpenCL_INCLUDE_DIR "${OPENCL_INCLUDE_DIR}" CACHE PATH "OpenCL include dir" FORCE)
set(OpenCL_INCLUDE_DIRS "${OPENCL_INCLUDE_DIR}" CACHE PATH "OpenCL include dirs" FORCE)
set(OPENCL_FOUND TRUE CACHE BOOL "OpenCL found" FORCE)
set(OPENCL_INCLUDE_DIRS "${OPENCL_INCLUDE_DIR}" CACHE PATH "OpenCL include dirs" FORCE)

# OpenCL library - handle Android vs host builds differently
# For Android: use extracted device library as stub for link-time, resolve at runtime
# For host: use system OpenCL (framework on macOS, library on Linux)
if(DEFINED OPENCL_LIBRARIES)
    # Library provided via CMake args (extracted from device by deps_manager.py)
    # This is used as a stub for Android cross-compilation
    message(STATUS "Using provided OpenCL library: ${OPENCL_LIBRARIES}")
    set(OpenCL_LIBRARY "${OPENCL_LIBRARIES}" CACHE STRING "OpenCL library" FORCE)
    set(OpenCL_LIBRARIES "${OPENCL_LIBRARIES}" CACHE STRING "OpenCL libraries" FORCE)
elseif(ANDROID)
    # For Android, OpenCL must be extracted from device (handled by porting orchestrator)
    message(FATAL_ERROR "OPENCL_LIBRARIES not provided. Run porting orchestrator to extract libOpenCL.so from device.")
else()
    # Native build - use system OpenCL
    if(APPLE)
        # On macOS, use the framework for linking (but our Khronos headers for includes)
        set(OpenCL_LIBRARY "-framework OpenCL" CACHE STRING "OpenCL library" FORCE)
        set(OpenCL_LIBRARIES "-framework OpenCL" CACHE STRING "OpenCL libraries" FORCE)
    else()
        set(OpenCL_LIBRARY "OpenCL" CACHE STRING "OpenCL library" FORCE)
        set(OpenCL_LIBRARIES "OpenCL" CACHE STRING "OpenCL libraries" FORCE)
    endif()
endif()

# Create OpenCL::OpenCL imported target BEFORE CLBlast is fetched
# This target uses our Khronos headers (with OpenCL/ symlink for macOS compatibility)
if(NOT TARGET OpenCL::OpenCL)
    add_library(OpenCL::OpenCL INTERFACE IMPORTED GLOBAL)
    set_target_properties(OpenCL::OpenCL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OPENCL_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OpenCL_LIBRARY}"
    )
    message(STATUS "Created OpenCL::OpenCL target with Khronos headers")
endif()

# Include our Khronos headers for all project code (BEFORE ensures priority over system paths)
include_directories(BEFORE ${OPENCL_INCLUDE_DIR})

# CLBlast — prefer pre-built from ~/.nnopt/deps/clblast/, fall back to FetchContent.
# Under fp16, SKIP the pre-built artifact: it's typically built without HALF=ON
# (Hgemm symbol absent), and a silent Sgemm fallback would defeat the whole
# point of the fp16 build. Always FetchContent + rebuild with HALF=ON for fp16.
set(NNOPT_CLBLAST_DIR "$ENV{HOME}/.nnopt/deps/clblast")
set(_NNOPT_USE_PREBUILT_CLBLAST FALSE)
if(NOT NNOPT_DTYPE STREQUAL "fp16"
   AND EXISTS "${NNOPT_CLBLAST_DIR}/lib/libclblast.a"
   AND EXISTS "${NNOPT_CLBLAST_DIR}/include/clblast.h")
    set(_NNOPT_USE_PREBUILT_CLBLAST TRUE)
endif()
if(_NNOPT_USE_PREBUILT_CLBLAST)
    message(STATUS "Using pre-built CLBlast from ${NNOPT_CLBLAST_DIR}")
    add_library(clblast STATIC IMPORTED GLOBAL)
    set_target_properties(clblast PROPERTIES
        IMPORTED_LOCATION "${NNOPT_CLBLAST_DIR}/lib/libclblast.a"
        INTERFACE_INCLUDE_DIRECTORIES "${NNOPT_CLBLAST_DIR}/include"
    )
    include_directories("${NNOPT_CLBLAST_DIR}/include")
else()
    if(NNOPT_DTYPE STREQUAL "fp16")
        message(STATUS "fp16 build — fetching CLBlast from source with HALF=ON")
    else()
        message(STATUS "Pre-built CLBlast not found — fetching from source (first build only)...")
    endif()
    include(FetchContent)
    FetchContent_Declare(
        clblast
        GIT_REPOSITORY https://github.com/CNugteren/CLBlast.git
        GIT_TAG 1.6.3
        GIT_SHALLOW TRUE
    )
    # On Android the canonical pattern is: build CLBlast as a shared lib,
    # deploy libclblast.so to $REMOTE_DIR/lib/, set LD_LIBRARY_PATH at run.
    # Deploy auto-pushes any *.so under build/<dtype>/_deps/*-build/ to
    # $REMOTE_DIR/lib/ — and run_android.sh prepends $REMOTE_DIR/lib to
    # LD_LIBRARY_PATH. So shared-link "just works". On non-Android hosts
    # we keep static-link to avoid runtime-loader path concerns.
    if(ANDROID)
        set(BUILD_SHARED_LIBS ON CACHE BOOL "Android: build CLBlast shared (deployed via Deploy)" FORCE)
    else()
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Host: link CLBlast statically" FORCE)
    endif()
    set(SAMPLES OFF CACHE BOOL "Disable samples" FORCE)
    set(TUNERS OFF CACHE BOOL "Disable tuners" FORCE)
    set(CLIENTS OFF CACHE BOOL "Disable clients" FORCE)
    set(TESTS OFF CACHE BOOL "Disable tests" FORCE)
    set(NETLIB OFF CACHE BOOL "Disable netlib" FORCE)
    FetchContent_MakeAvailable(clblast)
    if(ANDROID AND TARGET clblast)
        target_compile_options(clblast PRIVATE "-include" "sstream")
    endif()
    include_directories(${clblast_SOURCE_DIR}/include)
endif()

add_definitions(-DUSE_CLBLAST)
set(BACKEND_LIBRARIES clblast)
message(STATUS "CLBlast configured")

# Android-specific linker configuration for runtime OpenCL resolution
if(ANDROID)
    # Link against the extracted libOpenCL.so as a stub
    # Actual OpenCL symbols will be resolved at runtime on device
    list(APPEND BACKEND_LIBRARIES ${OPENCL_LIBRARIES})
    message(STATUS "Configuring Android linker for runtime OpenCL resolution...")
endif()


# ──────────────────────────────────────────────────────────────────────
# stb_image / stb_image_write — single-header image I/O for VLM ports.
# Auto-vendored on first build via FetchContent; populates
# src/third_party/{stb_image,stb_image_write}.h into the build tree and
# adds the directory to the include path. Headers are only #included by
# load_image.h / write_image.h (scaffolded for VLM ports only); text-only
# ports compile cleanly without ever pulling these symbols.
#
# The matching src/third_party/stb_image_impl.cpp (scaffolded alongside
# the headers) defines STB_IMAGE_IMPLEMENTATION + STB_IMAGE_WRITE_IMPLEMENTATION
# in exactly one TU so the headers can be #included header-style elsewhere.
# ──────────────────────────────────────────────────────────────────────
if(EXISTS "${CMAKE_SOURCE_DIR}/src/third_party/stb_image_impl.cpp")
    include(FetchContent)
    FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(stb)
    # stb is header-only — just expose its source dir on the include path.
    include_directories(${stb_SOURCE_DIR})
    message(STATUS "stb_image headers fetched from ${stb_SOURCE_DIR}")
endif()

# Layer sources — explicitly listed (updated by Build tool when new files appear)
set(LAYER_SOURCES
)

# Graph-mode op sources — agent appends one src/ops/<class>_<id>.cpp per
# ported node via PortNode. Glob picks them up automatically without the
# scaffolder needing to know about them at scaffold time. Empty in old-mode
# scaffolds (src/ops/ won't exist) — the glob silently no-ops.
file(GLOB GRAPH_OP_SOURCES "${CMAKE_SOURCE_DIR}/src/ops/*.cpp")

# Submodel sources — VLM/ASR/TTS ports use src/submodels/<role>/*.cpp
# (e.g. vision_tower/, projector/, lm/, audio_encoder/, text_decoder/).
# Glob silently no-ops for single-submodel text ports where the directory
# does not exist.
file(GLOB_RECURSE SUBMODEL_SOURCES "${CMAKE_SOURCE_DIR}/src/submodels/*.cpp")

# Third-party single-header impl sources — VLMs vendor stb_image[_write].h
# via FetchContent (declared in the backend section); the matching .cpp
# under src/third_party/ defines STB_IMAGE_IMPLEMENTATION so the headers
# emit code in exactly one translation unit.
file(GLOB THIRD_PARTY_SOURCES "${CMAKE_SOURCE_DIR}/src/third_party/*.cpp")

# Top-level src/*.cpp — globbed (not enumerated) so any helper the scaffold
# emits at src/ level gets compiled automatically without an explicit add.
# This covers main.cpp, model.cpp, sampler.cpp, opencl_context.cpp,
# weights.cpp, tokenizer.cpp, utils.cpp, plus any architecture-conditional
# helper the scaffold writes (e.g. rotary_tables.cpp for RoPE-based models,
# selective_scan_tables.cpp for SSM-based models). The glob is non-recursive
# so src/ops/, src/submodels/, src/third_party/ remain handled by their
# dedicated globs above.
file(GLOB CORE_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
# Exclude standalone probe binaries (each defines its own main()).
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/probe_recordable\\.cpp$")

# Source files
set(SOURCES
    ${CORE_SOURCES}
    ${LAYER_SOURCES}
    ${GRAPH_OP_SOURCES}
    ${SUBMODEL_SOURCES}
    ${THIRD_PARTY_SOURCES}
)

# Main inference binary
add_executable(mms_tts_eng_inference ${SOURCES})
target_include_directories(mms_tts_eng_inference PRIVATE
    src/
    src/layers/
    src/submodels/
    src/third_party/
)
# Suffix the binary with NNOPT_BIN_SUFFIX so fp32 (<name>_inference) and fp16
# (<name>_inference_fp16) coexist on the device side-by-side.
set_target_properties(mms_tts_eng_inference PROPERTIES
    OUTPUT_NAME "mms_tts_eng_inference${NNOPT_BIN_SUFFIX}"
)

# Link libraries (backend-specific)
target_link_libraries(mms_tts_eng_inference ${BACKEND_LIBRARIES})

# Optional standalone probe for cl_qcom_recordable_queues — built as a tiny
# binary alongside the main one so we can validate the extension's function
# pointers on this device before refactoring vocoder dispatch to use it.
add_executable(probe_recordable src/probe_recordable.cpp)
target_link_libraries(probe_recordable ${BACKEND_LIBRARIES})

# OpenMP path explored and rejected: on Android NDK 26 with libomp.so the
# parallel-for overhead was higher than the savings even with a work-size
# threshold (`if(OC*T > 50000)`). The runtime adds non-trivial per-call
# bookkeeping even when running serially. Single-threaded host convs are
# faster for the typical T_frames=110 path. Kept the host loop structure
# clean so a manual std::thread pool can be slotted in later if needed.

# Android-specific linker flags for runtime OpenCL resolution
if(ANDROID)
    target_link_options(mms_tts_eng_inference PRIVATE
        "-Wl,--allow-shlib-undefined"
        "-Wl,--unresolved-symbols=ignore-all"
        # CRITICAL: device-extracted libOpenCL.so stubs (cached in
        # ~/.nnopt/deps/opencl/lib/) carry DT_NEEDED entries pointing at
        # vendor GL libs (e.g. libGLES_mali.so on Mali, libCB.so on
        # Adreno-X). The linker by default copies those DT_NEEDEDs into
        # OUR executable, so a binary built from a Mali-extracted stub
        # then refuses to launch on Adreno (libGLES_mali.so missing) and
        # vice-versa. --no-copy-dt-needed-entries blocks the propagation
        # — we still resolve OpenCL symbols against the stub at link
        # time, but the DT_NEEDED on it doesn't cascade.
        "-Wl,--no-copy-dt-needed-entries"
        "-Wl,-rpath,/vendor/lib64:/vendor/lib:/system/lib64:/system/lib"
    )
    set_target_properties(mms_tts_eng_inference PROPERTIES
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
install(TARGETS mms_tts_eng_inference DESTINATION bin)

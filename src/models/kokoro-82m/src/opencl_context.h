#pragma once
#include <string>
#include <vector>

// Define OpenCL version BEFORE including headers to avoid version conflicts
#define CL_TARGET_OPENCL_VERSION 120

// Always use portable Khronos headers (works for cross-compilation to Android)
#include <CL/cl.h>

// ── cl_qcom_recordable_queues extension types ────────────────────────────────
// Not in stock cl_ext.h; declared inline so the entry points can be dlsym'd at
// runtime. Recipe validated on this device (Adreno 620) in the musicgen port:
// CL_QUEUE_RECORDABLE_QCOM = bit 30 ALONE on clCreateCommandQueue (combining
// with PROFILING_ENABLE fails); replay measured 4.04× cheaper per dispatch
// than live enqueue. Only clEnqueueNDRangeKernel can be recorded — no
// Fill/Copy/Read/WriteBuffer inside a recording.
typedef void* cl_recording_qcom;
struct cl_array_arg_qcom {
    cl_kernel    kernel;
    cl_uint      arg_indx;
    size_t       arg_size;
    const void*  arg_value;
};
struct cl_array_kernel_exec_info_qcom {
    cl_kernel       kernel;
    cl_uint         indx;
    size_t          param_value_size;
    const void*     param_value;
};
typedef cl_recording_qcom (CL_API_CALL *clNewRecordingQCOM_fn)(cl_command_queue, cl_int*);
typedef cl_int (CL_API_CALL *clEndRecordingQCOM_fn)(cl_recording_qcom);
typedef cl_int (CL_API_CALL *clReleaseRecordingQCOM_fn)(cl_recording_qcom);
typedef cl_int (CL_API_CALL *clEnqueueRecordingQCOM_fn)(
    cl_command_queue queue,
    cl_recording_qcom recording,
    size_t num_args,
    const cl_array_arg_qcom* args,
    size_t num_global_offsets,
    const cl_array_kernel_exec_info_qcom* global_offsets,
    size_t num_global_work_sizes,
    const cl_array_kernel_exec_info_qcom* global_work_sizes,
    size_t num_local_work_sizes,
    const cl_array_kernel_exec_info_qcom* local_work_sizes,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

class OpenCLContext {
public:
    OpenCLContext();
    ~OpenCLContext();

    bool initialize(int platform_idx = 0, int device_idx = 0);
    cl_program build_program(const std::string& source, const std::string& options = "");
    cl_program build_program_from_file(const std::string& path, const std::string& options = "");

    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
    cl_device_id device() const { return device_; }

    // Device info
    std::string device_name() const;
    size_t max_work_group_size() const;
    size_t local_mem_size() const;

    // ── cl_qcom_recordable_queues helpers (entry points dlsym'd at init) ────
    // A queue created with CL_QUEUE_RECORDABLE_QCOM alone does NOT execute its
    // dispatches — they are captured into a recording (between new_recording /
    // end_recording) which is then replayed on the LIVE queue via
    // enqueue_recording.
    bool has_recordable_queues() const { return record_fns_loaded_; }
    cl_command_queue create_recordable_queue();   // nullptr on failure
    cl_recording_qcom new_recording(cl_command_queue q) const;
    cl_int end_recording(cl_recording_qcom rec) const;
    cl_int release_recording(cl_recording_qcom rec) const;
    cl_int enqueue_recording(cl_command_queue live_q, cl_recording_qcom rec,
                             size_t num_args, const cl_array_arg_qcom* args) const;

private:
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;

    bool record_fns_loaded_ = false;
    clNewRecordingQCOM_fn       fn_new_recording_     = nullptr;
    clEndRecordingQCOM_fn       fn_end_recording_     = nullptr;
    clReleaseRecordingQCOM_fn   fn_release_recording_ = nullptr;
    clEnqueueRecordingQCOM_fn   fn_enqueue_recording_ = nullptr;
};

// ─── Cached OpenCL program build ────────────────────────────────────────
// Compile the source on first run, save the compiled binary to
// $NNOPT_KERNEL_CACHE_DIR/<cache_key>.<source_hash>.bin (defaults to
// "kernel_cache" in CWD), and on subsequent runs load that binary via
// clCreateProgramWithBinary. Saves ~5 s of cold-start cost on Adreno
// across the ~10 separately-built programs in this codebase.
//
// Cache invalidation: the file name includes a hash of (source + options
// + device name), so any change to kernel text or build flags or running
// on a different device triggers a recompile + cache refresh.
//
// Args:
//   ctx        — OpenCL context for clCreateProgram*
//   dev        — single target device
//   source     — kernel source string (NUL-terminated C string)
//   options    — clBuildProgram options (NUL-terminated; "" or nullptr OK)
//   cache_key  — short stable id for this program (e.g. "decoder", "bert").
//                Just used in the cache file name for human-readability.
//   out_err    — optional: receives the cl_int from the underlying call
//
// Returns: cl_program on success (caller owns; release with clReleaseProgram),
//          nullptr on build/link failure. The full BuildLog is logged via
//          NNOPT_ERROR_FMT on failure.
cl_program nnopt_build_program_cached(cl_context ctx,
                                       cl_device_id dev,
                                       const char* source,
                                       const char* options,
                                       const char* cache_key,
                                       cl_int* out_err = nullptr);

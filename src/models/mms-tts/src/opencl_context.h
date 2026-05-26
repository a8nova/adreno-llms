#pragma once
#include <string>
#include <vector>

// Define OpenCL version BEFORE including headers to avoid version conflicts
#define CL_TARGET_OPENCL_VERSION 120

// Always use portable Khronos headers (works for cross-compilation to Android)
#include <CL/cl.h>

// ── cl_qcom_recordable_queues extension types (Adreno OpenCL Guide §9.1.3).
//   Not in stock cl_ext.h; declared inline so we can dlsym the entry points
//   at runtime. Struct layouts match MNN's vendored cl_ext_qcom.h (the only
//   public source of these layouts — Qualcomm's official header isn't in
//   Khronos's repo).
//
//   The CRITICAL detail every prior attempt has gotten wrong: the first
//   field of each per-replay-override struct is `cl_uint dispatch_index`,
//   NOT a cl_kernel pointer. dispatch_index is the 0-based index into the
//   recorded sequence (the Nth clEnqueueNDRangeKernel call you made during
//   recording). A wrong first field returns CL_INVALID_OPERATION (-59).
typedef void* cl_recording_qcom;

struct cl_array_arg_qcom {
    cl_uint      dispatch_index;
    cl_uint      arg_index;
    size_t       arg_size;
    const void*  arg_value;
};
struct cl_workgroup_qcom {
    cl_uint       dispatch_index;
    const size_t* workgroup_size;
};
struct cl_offset_qcom {
    cl_uint dispatch_index;
    size_t  offsets[3];
};

typedef cl_recording_qcom (CL_API_CALL *clNewRecordingQCOM_fn)(cl_command_queue, cl_int*);
typedef cl_int (CL_API_CALL *clEndRecordingQCOM_fn)(cl_recording_qcom);
typedef cl_int (CL_API_CALL *clReleaseRecordingQCOM_fn)(cl_recording_qcom);
typedef cl_int (CL_API_CALL *clEnqueueRecordingQCOM_fn)(
    cl_command_queue queue,
    cl_recording_qcom recording,
    size_t num_args,            const cl_array_arg_qcom*  args,
    size_t num_global_offsets,  const cl_offset_qcom*     global_offsets,
    size_t num_global_wg_sizes, const cl_workgroup_qcom*  global_work_sizes,
    size_t num_local_wg_sizes,  const cl_workgroup_qcom*  local_work_sizes,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

// CL_QUEUE_RECORDABLE_QCOM. Per Adreno guide §9.1.3, this is `(1u << 30)`.
// MUST be passed ALONE in cl_queue_properties — OR'ing with
// CL_QUEUE_PROFILING_ENABLE returns CL_INVALID_VALUE (-30).
//
// CL_QUEUE_PROPERTIES = 0x1093 — the property key name in
// clCreateCommandQueueWithProperties.
constexpr cl_command_queue_properties NNOPT_CL_QUEUE_RECORDABLE_QCOM =
    (cl_command_queue_properties)1 << 30;
constexpr cl_uint NNOPT_CL_QUEUE_PROPERTIES = 0x1093;

class OpenCLContext {
public:
    OpenCLContext();
    ~OpenCLContext();

    bool initialize(int platform_idx = 0, int device_idx = 0);
    cl_program build_program(const std::string& source, const std::string& options = "");
    cl_program build_program_from_file(const std::string& path, const std::string& options = "");

    // Program cache (build once, reuse everywhere)
    bool ensure_program_from_file(const std::string& path, const std::string& options = "");
    cl_program get_program(const std::string& path) const;

    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
    cl_device_id device() const { return device_; }

    // Device info
    std::string device_name() const;
    size_t max_work_group_size() const;
    size_t local_mem_size() const;

    // ── cl_qcom_recordable_queues helpers (loaded via dlsym at init).
    //   Use create_recordable_queue() to build a queue you can record into,
    //   then new_recording → ... dispatch kernels ... → end_recording.
    //   Replay on the LIVE queue (queue()) via enqueue_recording with the
    //   per-kernel arg updates packed into cl_array_arg_qcom[].
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

    struct ProgramEntry {
        std::string path;
        cl_program program;
    };
    std::vector<ProgramEntry> program_cache_;
};
